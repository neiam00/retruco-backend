package api

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const turnstileVerifyURL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

var errTurnstileRejected = errors.New("turnstile rejected registration")

type TurnstileConfig struct {
	SiteKey     string
	SecretKey   string
	Client      *http.Client
	Development bool
}

func (s *Server) registrationConfig(w http.ResponseWriter, r *http.Request) {
	if s.config.Turnstile.SiteKey == "" || s.config.Turnstile.SecretKey == "" {
		fail(w, http.StatusServiceUnavailable, "El registro no está disponible en este momento.")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"siteKey": s.config.Turnstile.SiteKey})
}

func (s *Server) verifyRegistration(ctx context.Context, token string) error {
	if token == "" || len(token) > 2048 {
		return errTurnstileRejected
	}
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	form := url.Values{"secret": {s.config.Turnstile.SecretKey}, "response": {token}}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, turnstileVerifyURL, strings.NewReader(form.Encode()))
	if err != nil {
		return fmt.Errorf("create turnstile request: %w", err)
	}
	request.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	response, err := s.turnstileClient.Do(request)
	if err != nil {
		return fmt.Errorf("verify turnstile: %w", err)
	}
	defer func() {
		if err := response.Body.Close(); err != nil {
			s.log.Warn("close turnstile response", "error", err)
		}
	}()
	if response.StatusCode != http.StatusOK {
		return fmt.Errorf("turnstile returned HTTP %d", response.StatusCode)
	}
	var result struct {
		Success  bool   `json:"success"`
		Hostname string `json:"hostname"`
		Action   string `json:"action"`
	}
	if err := json.NewDecoder(io.LimitReader(response.Body, 16384)).Decode(&result); err != nil {
		return fmt.Errorf("decode turnstile response: %w", err)
	}
	expectedAction := "register"
	// Cloudflare's documented local test response uses action "test". Accept it
	// only with the exact public dummy key pair in explicitly enabled development.
	if s.config.Turnstile.Development &&
		s.config.Turnstile.SiteKey == "1x00000000000000000000AA" &&
		s.config.Turnstile.SecretKey == "1x0000000000000000000000000000000AA" {
		expectedAction = "test"
	}
	if !result.Success || result.Action != expectedAction {
		return errTurnstileRejected
	}
	// Use configured public frontend origins, never a client-supplied hostname.
	for _, origin := range s.config.Origins {
		parsed, err := url.Parse(origin)
		if err == nil && parsed.Hostname() != "" && strings.EqualFold(parsed.Hostname(), result.Hostname) {
			return nil
		}
	}
	return errTurnstileRejected
}
