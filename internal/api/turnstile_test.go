package api_test

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/neiam00/retruco-backend/internal/api"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/crypto/bcrypt"
)

type turnstileTransport func(*http.Request) (*http.Response, error)

func (f turnstileTransport) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

type registrationStore struct {
	api.Repository
	users    int
	sessions int
}

func (s *registrationStore) CreateUser(_ context.Context, _ store.User, _, _ string) error {
	s.users++
	return nil
}

func (s *registrationStore) CreateSession(_ context.Context, _, _ string, _ time.Time) error {
	s.sessions++
	return nil
}

func TestRegistrationRequiresVerifiedTurnstile(t *testing.T) {
	for _, scenario := range []struct {
		name        string
		token       string
		response    string
		status      int
		wantStatus  int
		wantCalls   int
		dummyKeys   bool
		development bool
	}{
		{name: "valid", token: "valid-token", response: `{"success":true,"hostname":"cards.example.test","action":"register"}`, status: 200, wantStatus: 200, wantCalls: 1},
		{name: "development test keys", token: "XXXX.DUMMY.TOKEN.XXXX", response: `{"success":true,"hostname":"example.com"}`, status: 200, wantStatus: 200, wantCalls: 1, dummyKeys: true, development: true},
		{name: "production rejects test action", token: "XXXX.DUMMY.TOKEN.XXXX", response: `{"success":true,"hostname":"example.com"}`, status: 200, wantStatus: 403, wantCalls: 1, dummyKeys: true},
		{name: "development real keys require register action", token: "token", response: `{"success":true,"hostname":"cards.example.test","action":"test"}`, status: 200, wantStatus: 403, wantCalls: 1, development: true},
		{name: "missing token", wantStatus: 403},
		{name: "oversized token", token: strings.Repeat("a", 2049), wantStatus: 403},
		{name: "expired or reused", token: "old-token", response: `{"success":false,"error-codes":["timeout-or-duplicate"]}`, status: 200, wantStatus: 403, wantCalls: 1},
		{name: "wrong hostname", token: "token", response: `{"success":true,"hostname":"attacker.example","action":"register"}`, status: 200, wantStatus: 403, wantCalls: 1},
		{name: "wrong action", token: "token", response: `{"success":true,"hostname":"cards.example.test","action":"login"}`, status: 200, wantStatus: 403, wantCalls: 1},
		{name: "empty response", token: "token", response: `{}`, status: 200, wantStatus: 403, wantCalls: 1},
		{name: "malformed response", token: "token", response: `broken`, status: 200, wantStatus: 503, wantCalls: 1},
		{name: "service error", token: "token", response: `{}`, status: 500, wantStatus: 503, wantCalls: 1},
		{name: "network error", token: "token", wantStatus: 503, wantCalls: 1},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			calls := 0
			config := api.TurnstileConfig{SiteKey: "public-test-key", SecretKey: "private-test-secret", Development: scenario.development}
			if scenario.dummyKeys {
				config.SiteKey = "1x00000000000000000000AA"
				config.SecretKey = "1x0000000000000000000000000000000AA"
			}
			client := &http.Client{Transport: turnstileTransport(func(request *http.Request) (*http.Response, error) {
				calls++
				assert.Equal(t, "https://challenges.cloudflare.com/turnstile/v0/siteverify", request.URL.String())
				assert.Equal(t, http.MethodPost, request.Method)
				require.NoError(t, request.ParseForm())
				assert.Equal(t, config.SecretKey, request.Form.Get("secret"))
				assert.Equal(t, scenario.token, request.Form.Get("response"))
				deadline, ok := request.Context().Deadline()
				assert.True(t, ok)
				assert.LessOrEqual(t, time.Until(deadline), 5*time.Second)
				if scenario.status == 0 {
					return nil, errors.New("verification offline")
				}
				return &http.Response{StatusCode: scenario.status, Body: io.NopCloser(strings.NewReader(scenario.response)), Header: http.Header{}}, nil
			})}
			config.Client = client
			repository := &registrationStore{}
			server, err := api.New(t.Context(), api.Config{
				Origins:   []string{"https://cards.example.test", "http://localhost:5173"},
				Turnstile: config,
			}, repository, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
			require.NoError(t, err)
			body, err := json.Marshal(map[string]string{"username": "player", "password": "valid-password", "turnstileToken": scenario.token})
			require.NoError(t, err)
			request := httptest.NewRequest(http.MethodPost, "/api/register", strings.NewReader(string(body)))
			request.Header.Set("Origin", "https://cards.example.test")
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			server.Handler().ServeHTTP(response, request)
			require.Equal(t, scenario.wantStatus, response.Code, response.Body.String())
			assert.Equal(t, scenario.wantCalls, calls)
			if scenario.wantStatus == http.StatusOK {
				assert.Equal(t, 1, repository.users)
				assert.Equal(t, 1, repository.sessions)
				assert.NotEmpty(t, response.Header().Get("Set-Cookie"))
			} else {
				assert.Zero(t, repository.users)
				assert.Zero(t, repository.sessions)
				assert.Empty(t, response.Header().Get("Set-Cookie"))
			}
		})
	}
}

func TestTurnstileConfigExposesOnlyPublicKeyAndMissingKeysFailClosed(t *testing.T) {
	for _, scenario := range []struct {
		name   string
		config api.TurnstileConfig
		status int
	}{
		{name: "configured", config: api.TurnstileConfig{SiteKey: "public-key", SecretKey: "secret-key"}, status: 200},
		{name: "missing both", status: 503},
		{name: "missing secret", config: api.TurnstileConfig{SiteKey: "public-key"}, status: 503},
		{name: "missing site key", config: api.TurnstileConfig{SecretKey: "secret-key"}, status: 503},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			server, err := api.New(t.Context(), api.Config{Origins: []string{"https://cards.example.test"}, Turnstile: scenario.config}, nil, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
			require.NoError(t, err)
			handler := server.Handler()
			response := httptest.NewRecorder()
			handler.ServeHTTP(response, httptest.NewRequest(http.MethodGet, "/api/turnstile", nil))
			require.Equal(t, scenario.status, response.Code)
			assert.NotContains(t, response.Body.String(), "secret-key")
			assert.Equal(t, "no-store", response.Header().Get("Cache-Control"))
			if scenario.status == 200 {
				assert.JSONEq(t, `{"siteKey":"public-key"}`, response.Body.String())
				legacy := httptest.NewRecorder()
				handler.ServeHTTP(legacy, httptest.NewRequest(http.MethodGet, "/api/registration", nil))
				require.Equal(t, http.StatusOK, legacy.Code)
				assert.JSONEq(t, `{"siteKey":"public-key"}`, legacy.Body.String())
				return
			}
			request := httptest.NewRequest(http.MethodPost, "/api/register", strings.NewReader(`{"username":"player","password":"valid-password","turnstileToken":"anything"}`))
			request.Header.Set("Origin", "https://cards.example.test")
			request.Header.Set("Content-Type", "application/json")
			response = httptest.NewRecorder()
			handler.ServeHTTP(response, request)
			assert.Equal(t, http.StatusServiceUnavailable, response.Code)
		})
	}
}

type loginStore struct {
	registrationStore
	passwordHash string
	lookups      int
}

func (s *loginStore) Credentials(_ context.Context, _ string) (store.User, string, error) {
	s.lookups++
	return store.User{ID: "player-id", Name: "player", Registered: true}, s.passwordHash, nil
}

func TestLoginRequiresVerifiedTurnstileBeforeCheckingCredentials(t *testing.T) {
	hash, err := bcrypt.GenerateFromPassword([]byte("correct-password"), bcrypt.MinCost)
	require.NoError(t, err)
	for _, scenario := range []struct {
		name        string
		token       string
		response    string
		password    string
		unavailable bool
		missingKeys bool
		wantStatus  int
		wantLookups int
	}{
		{name: "missing token", wantStatus: 403},
		{name: "expired or reused token", token: "old-token", response: `{"success":false,"error-codes":["timeout-or-duplicate"]}`, wantStatus: 403},
		{name: "registration token cannot log in", token: "register-token", response: `{"success":true,"hostname":"cards.example.test","action":"register"}`, wantStatus: 403},
		{name: "wrong hostname", token: "token", response: `{"success":true,"hostname":"attacker.example","action":"login"}`, wantStatus: 403},
		{name: "verification unavailable", token: "token", unavailable: true, wantStatus: 503},
		{name: "missing keys", token: "token", missingKeys: true, wantStatus: 503},
		{name: "wrong password after verification", token: "valid-token", response: `{"success":true,"hostname":"cards.example.test","action":"login"}`, password: "wrong-password", wantStatus: 401, wantLookups: 1},
		{name: "verified login", token: "valid-token", response: `{"success":true,"hostname":"cards.example.test","action":"login"}`, wantStatus: 200, wantLookups: 1},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			repository := &loginStore{passwordHash: string(hash)}
			client := &http.Client{Transport: turnstileTransport(func(request *http.Request) (*http.Response, error) {
				require.NoError(t, request.ParseForm())
				assert.Equal(t, scenario.token, request.Form.Get("response"))
				if scenario.unavailable {
					return nil, errors.New("verification offline")
				}
				return &http.Response{StatusCode: 200, Body: io.NopCloser(strings.NewReader(scenario.response))}, nil
			})}
			config := api.TurnstileConfig{SiteKey: "public-key", SecretKey: "secret-key", Client: client}
			if scenario.missingKeys {
				config = api.TurnstileConfig{}
			}
			server, err := api.New(t.Context(), api.Config{Origins: []string{"https://cards.example.test"}, Turnstile: config}, repository, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
			require.NoError(t, err)
			password := scenario.password
			if password == "" {
				password = "correct-password"
			}
			body := map[string]string{"username": "player", "password": password}
			if scenario.token != "" {
				body["turnstileToken"] = scenario.token
			}
			payload, err := json.Marshal(body)
			require.NoError(t, err)
			request := httptest.NewRequest(http.MethodPost, "/api/login", strings.NewReader(string(payload)))
			request.Header.Set("Origin", "https://cards.example.test")
			request.Header.Set("Content-Type", "application/json")
			response := httptest.NewRecorder()
			server.Handler().ServeHTTP(response, request)
			require.Equal(t, scenario.wantStatus, response.Code, response.Body.String())
			assert.Equal(t, scenario.wantLookups, repository.lookups)
			assert.Zero(t, repository.users)
			if scenario.wantStatus == 200 {
				assert.Equal(t, 1, repository.sessions)
				assert.NotEmpty(t, response.Header().Get("Set-Cookie"))
			} else {
				assert.Zero(t, repository.sessions)
				assert.Empty(t, response.Header().Get("Set-Cookie"))
			}
		})
	}
}
