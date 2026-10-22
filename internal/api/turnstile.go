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

var errTurnstileRejected = errors.New("turnstile rejected authentication")

type TurnstileConfig struct {
	SiteKey     string
	SecretKey   string
	Client      *http.Client
	Development bool
}

func (s *Server) turnstileConfig(w http.ResponseWriter, r *http.Request) {
	if s.config.Turnstile.SiteKey == "" || s.config.Turnstile.SecretKey == "" {
		fail(w, http.StatusServiceUnavailable, "La verificación de seguridad no está disponible en este momento.")
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"siteKey": s.config.Turnstile.SiteKey})
}

func (s *Server) requireTurnstile(w http.ResponseWriter, r *http.Request, token, action string) bool {
	if s.config.Turnstile.SiteKey == "" || s.config.Turnstile.SecretKey == "" {
		fail(w, http.StatusServiceUnavailable, "La verificación de seguridad no está disponible en este momento.")
		return false
	}
	if err := s.verifyTurnstile(r.Context(), token, action); err != nil {
		if errors.Is(err, errTurnstileRejected) {
			fail(w, http.StatusForbidden, "Completa de nuevo la verificación de seguridad.")
		} else {
			s.log.Error("authentication verification unavailable", "action", action, "error", err)
			fail(w, http.StatusServiceUnavailable, "No pudimos verificarte. Inténtalo de nuevo.")
		}
		return false
	}
	return true
}

func (s *Server) verifyTurnstile(ctx context.Context, token, action string) error {
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
	// Cloudflare's public dummy pair returns synthetic metadata, not the real
	// widget action/hostname. Only this exact pair and token in development use
	// its success flag alone; real keys and production always check metadata.
	if s.config.Turnstile.Development &&
		s.config.Turnstile.SiteKey == "1x00000000000000000000AA" &&
		s.config.Turnstile.SecretKey == "1x0000000000000000000000000000000AA" &&
		token == "XXXX.DUMMY.TOKEN.XXXX" && result.Success {
		return nil
	}
	if !result.Success || result.Action != action {
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
