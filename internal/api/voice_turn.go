package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
)

var (
	turnTokenIDPattern         = regexp.MustCompile(`^[A-Za-z0-9_-]{1,128}$`)
	errVoiceCredentialsLimited = errors.New("voice credential issuance rate exceeded")
)

func newVoiceClient(config VoiceConfig) (*http.Client, error) {
	if (config.TURNTokenID != "") != (config.TURNAPIToken != "") {
		return nil, errors.New("Cloudflare TURN Token ID and API token must be configured together")
	}
	if config.TURNTokenID != "" && !turnTokenIDPattern.MatchString(config.TURNTokenID) {
		return nil, errors.New("invalid Cloudflare TURN Token ID")
	}
	if strings.ContainsAny(config.TURNAPIToken, " \t\r\n") {
		return nil, errors.New("Cloudflare TURN API token must not contain whitespace")
	}
	client := http.Client{}
	if config.Client != nil {
		client = *config.Client
	}
	client.Timeout = 5 * time.Second
	client.CheckRedirect = func(_ *http.Request, _ []*http.Request) error {
		return http.ErrUseLastResponse
	}
	return &client, nil
}

func (s *Server) voiceICEServers(ctx context.Context, userID string) ([]iceServer, error) {
	if s.config.Voice.TURNTokenID == "" {
		return nil, nil // Local development can use direct peer connections.
	}
	// Joining and refreshing share a per-user budget, separate from signaling.
	if !s.allow("voice-credentials:"+userID, 4) {
		return nil, errVoiceCredentialsLimited
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	endpoint := "https://rtc.live.cloudflare.com/v1/turn/keys/" + s.config.Voice.TURNTokenID + "/credentials/generate-ice-servers"
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, strings.NewReader(`{"ttl":3600}`))
	if err != nil {
		return nil, errors.New("create Cloudflare TURN request failed")
	}
	request.Header.Set("Authorization", "Bearer "+s.config.Voice.TURNAPIToken)
	request.Header.Set("Content-Type", "application/json")
	response, err := s.voiceClient.Do(request)
	if err != nil {
		// Do not log URLs, headers, provider bodies, or transport error strings
		// that could contain credentials. Cancellation remains identifiable.
		if ctx.Err() != nil {
			return nil, fmt.Errorf("Cloudflare TURN request: %w", ctx.Err())
		}
		return nil, errors.New("Cloudflare TURN request failed")
	}
	defer func() { _ = response.Body.Close() }()
	if response.StatusCode != http.StatusCreated {
		return nil, fmt.Errorf("Cloudflare TURN returned HTTP %d", response.StatusCode)
	}
	const maxBody = 32 * 1024
	body, err := io.ReadAll(io.LimitReader(response.Body, maxBody+1))
	if err != nil || len(body) > maxBody {
		return nil, errors.New("Cloudflare TURN response could not be read")
	}
	var result struct {
		ICEServers []iceServer `json:"iceServers"`
	}
	if json.Unmarshal(body, &result) != nil {
		return nil, errors.New("invalid Cloudflare TURN response")
	}
	servers := make([]iceServer, 0, len(result.ICEServers))
	hasTURN := false
	for _, server := range result.ICEServers {
		urls := make([]string, 0, len(server.URLs))
		for _, raw := range server.URLs {
			parsed, err := url.Parse(raw)
			if err != nil || (parsed.Scheme != "stun" && parsed.Scheme != "turn" && parsed.Scheme != "turns") {
				return nil, errors.New("invalid Cloudflare ICE URL")
			}
			host, port, err := net.SplitHostPort(parsed.Opaque)
			if err != nil || host == "" || port == "" {
				return nil, errors.New("invalid Cloudflare ICE address")
			}
			if port == "53" {
				continue // Browsers block the alternate DNS port.
			}
			if parsed.Scheme != "stun" {
				if server.Username == "" || server.Credential == "" {
					return nil, errors.New("Cloudflare TURN response is missing credentials")
				}
				hasTURN = true
			}
			urls = append(urls, raw)
		}
		if len(urls) > 0 {
			server.URLs = urls
			servers = append(servers, server)
		}
	}
	if !hasTURN {
		return nil, errors.New("Cloudflare response contains no usable TURN server")
	}
	return servers, nil
}
