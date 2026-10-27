package api

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type voiceTransport func(*http.Request) (*http.Response, error)

func (f voiceTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

func TestVoiceConfigurationRequiresCloudflareCredentialPair(t *testing.T) {
	for _, tc := range []struct {
		name   string
		config VoiceConfig
		bad    bool
	}{
		{name: "direct local connections"},
		{name: "configured", config: VoiceConfig{TURNTokenID: "key-id", TURNAPIToken: "private-token"}},
		{name: "missing API token", config: VoiceConfig{TURNTokenID: "key-id"}, bad: true},
		{name: "missing ID", config: VoiceConfig{TURNAPIToken: "private-token"}, bad: true},
		{name: "invalid path", config: VoiceConfig{TURNTokenID: "../keys", TURNAPIToken: "private-token"}, bad: true},
		{name: "invalid header", config: VoiceConfig{TURNTokenID: "key-id", TURNAPIToken: "private-token\n"}, bad: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			_, err := newVoiceClient(tc.config)
			if tc.bad {
				require.Error(t, err)
				assert.NotContains(t, err.Error(), "private-token")
			} else {
				require.NoError(t, err)
			}
		})
	}
}

func TestVoiceCloudflareCredentialsAndIssuanceBudget(t *testing.T) {
	calls := 0
	config := VoiceConfig{TURNTokenID: "key-id", TURNAPIToken: "private-token"}
	config.Client = &http.Client{Transport: voiceTransport(func(r *http.Request) (*http.Response, error) {
		calls++
		assert.Equal(t, "https://rtc.live.cloudflare.com/v1/turn/keys/key-id/credentials/generate-ice-servers", r.URL.String())
		assert.Equal(t, http.MethodPost, r.Method)
		assert.Equal(t, "Bearer private-token", r.Header.Get("Authorization"))
		assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		assert.JSONEq(t, `{"ttl":3600}`, string(body))
		deadline, ok := r.Context().Deadline()
		assert.True(t, ok)
		assert.LessOrEqual(t, time.Until(deadline), 5*time.Second)
		return &http.Response{StatusCode: 201, Body: io.NopCloser(strings.NewReader(`{"iceServers":[
			{"urls":["stun:stun.cloudflare.com:3478","stun:stun.cloudflare.com:53"]},
			{"urls":["turn:turn.cloudflare.com:3478?transport=udp","turn:turn.cloudflare.com:53?transport=udp","turns:turn.cloudflare.com:443?transport=tcp"],"username":"temporary-user","credential":"temporary-password"}
		]}`))}, nil
	})}
	client, err := newVoiceClient(config)
	require.NoError(t, err)
	s := &Server{config: Config{Voice: config}, voiceClient: client, limits: map[string]window{}}
	for range 4 {
		servers, err := s.voiceICEServers(t.Context(), "ana")
		require.NoError(t, err)
		require.Len(t, servers, 2)
		assert.Equal(t, []string{"stun:stun.cloudflare.com:3478"}, servers[0].URLs)
		assert.Equal(t, []string{"turn:turn.cloudflare.com:3478?transport=udp", "turns:turn.cloudflare.com:443?transport=tcp"}, servers[1].URLs)
		assert.Equal(t, "temporary-user", servers[1].Username)
		assert.Equal(t, "temporary-password", servers[1].Credential)
		wire, err := json.Marshal(servers)
		require.NoError(t, err)
		assert.NotContains(t, string(wire), config.TURNAPIToken)
	}
	_, err = s.voiceICEServers(t.Context(), "ana")
	require.ErrorIs(t, err, errVoiceCredentialsLimited)
	assert.Equal(t, 4, calls)
	_, err = s.voiceICEServers(t.Context(), "beto")
	require.NoError(t, err)
	assert.Equal(t, 5, calls)
}

func TestVoiceCloudflareRejectsFailedOrUnusableResponses(t *testing.T) {
	for _, tc := range []struct {
		name   string
		status int
		body   string
	}{
		{name: "unauthorized", status: 401, body: "private-token"},
		{name: "provider unavailable", status: 503},
		{name: "redirect", status: 302},
		{name: "invalid JSON", status: 201, body: "private-token"},
		{name: "empty list", status: 201, body: `{"iceServers":[]}`},
		{name: "STUN only", status: 201, body: `{"iceServers":[{"urls":["stun:stun.cloudflare.com:3478"]}]}`},
		{name: "missing credentials", status: 201, body: `{"iceServers":[{"urls":["turn:turn.cloudflare.com:3478"]}]}`},
		{name: "invalid URL", status: 201, body: `{"iceServers":[{"urls":["https://example.com"]}]}`},
		{name: "oversized body", status: 201, body: strings.Repeat(" ", 32769)},
		{name: "network failure"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			config := VoiceConfig{TURNTokenID: "key-id", TURNAPIToken: "private-token"}
			config.Client = &http.Client{Transport: voiceTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				if tc.status == 0 {
					return nil, errors.New("transport accidentally includes private-token")
				}
				return &http.Response{StatusCode: tc.status, Header: http.Header{"Location": {"https://example.com/redirect"}}, Body: io.NopCloser(strings.NewReader(tc.body))}, nil
			})}
			client, err := newVoiceClient(config)
			require.NoError(t, err)
			s := &Server{config: Config{Voice: config}, voiceClient: client, limits: map[string]window{}}
			servers, err := s.voiceICEServers(t.Context(), "ana")
			require.Error(t, err)
			assert.Empty(t, servers)
			assert.NotContains(t, err.Error(), "private-token")
			assert.Equal(t, 1, calls, "redirects must not forward the token or issue another request")
		})
	}
}

func TestVoiceCloudflareRequestHonorsCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(t.Context())
	config := VoiceConfig{TURNTokenID: "key-id", TURNAPIToken: "private-token"}
	config.Client = &http.Client{Transport: voiceTransport(func(r *http.Request) (*http.Response, error) {
		cancel()
		<-r.Context().Done()
		return nil, r.Context().Err()
	})}
	client, err := newVoiceClient(config)
	require.NoError(t, err)
	s := &Server{config: Config{Voice: config}, voiceClient: client, limits: map[string]window{}}
	_, err = s.voiceICEServers(ctx, "ana")
	require.ErrorIs(t, err, context.Canceled)
}
