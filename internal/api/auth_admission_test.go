package api_test

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/neiam00/retruco-backend/internal/api"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/crypto/bcrypt"
)

type blockingPasswordRepository struct {
	api.Repository
	entered         chan struct{}
	release         chan struct{}
	hash            string
	firstError      error
	credentialCalls atomic.Int32
	sessionWrites   atomic.Int32
	userWrites      atomic.Int32
}

func (r *blockingPasswordRepository) Credentials(ctx context.Context, _ string) (store.User, string, error) {
	if r.credentialCalls.Add(1) == 1 {
		close(r.entered)
		select {
		case <-ctx.Done():
			return store.User{}, "", ctx.Err()
		case <-r.release:
		}
		if r.firstError != nil {
			return store.User{}, "", r.firstError
		}
	}
	return store.User{ID: "account", Name: "Account", Registered: true}, r.hash, nil
}

func (r *blockingPasswordRepository) CreateSession(context.Context, string, string, time.Time) error {
	r.sessionWrites.Add(1)
	return nil
}

func (r *blockingPasswordRepository) CreateUser(context.Context, store.User, string, string) error {
	r.userWrites.Add(1)
	return nil
}

func TestPasswordAdmissionProtectsOtherRequestsAndRecovers(t *testing.T) {
	for _, failure := range []bool{false, true} {
		name := "successful login"
		if failure {
			name = "storage failure"
		}
		t.Run(name, func(t *testing.T) {
			hash, err := bcrypt.GenerateFromPassword([]byte("correct-password"), bcrypt.MinCost)
			require.NoError(t, err)
			repository := &blockingPasswordRepository{entered: make(chan struct{}), release: make(chan struct{}), hash: string(hash)}
			if failure {
				repository.firstError = errors.New("storage unavailable")
			}
			t.Cleanup(func() {
				select {
				case <-repository.release:
				default:
					close(repository.release)
				}
			})
			server, err := api.New(t.Context(), api.Config{
				Origins: []string{"https://retruco.test"}, ProxySecret: "test-edge",
				Turnstile: api.TurnstileConfig{
					SiteKey: "1x00000000000000000000AA", SecretKey: "1x0000000000000000000000000000000AA", Development: true,
					Client: &http.Client{Transport: turnstileTransport(func(*http.Request) (*http.Response, error) {
						return &http.Response{StatusCode: http.StatusOK, Body: io.NopCloser(strings.NewReader(`{"success":true}`))}, nil
					})},
				},
			}, repository, nil, slog.New(slog.NewTextHandler(io.Discard, nil)))
			require.NoError(t, err)
			httpServer := httptest.NewServer(server.Handler())
			t.Cleanup(httpServer.Close)
			client := &http.Client{Timeout: 2 * time.Second}
			request := func(method, path, ip string) (*http.Response, error) {
				body := `{"username":"account","password":"correct-password","turnstileToken":"XXXX.DUMMY.TOKEN.XXXX"}`
				req, err := http.NewRequestWithContext(t.Context(), method, httpServer.URL+path, strings.NewReader(body))
				if err != nil {
					return nil, err
				}
				req.Header.Set("Origin", "https://retruco.test")
				req.Header.Set("Content-Type", "application/json")
				req.Header.Set("X-Retruco-Proxy-Secret", "test-edge")
				req.Header.Set("X-Retruco-Client-IP", ip)
				// #nosec G704 -- Only this test's local httptest server receives requests.
				return client.Do(req)
			}
			type result struct {
				response *http.Response
				err      error
			}
			first := make(chan result, 1)
			go func() {
				response, err := request(http.MethodPost, "/api/login", "192.0.2.1")
				first <- result{response: response, err: err}
			}()
			select {
			case <-repository.entered:
			case <-time.After(2 * time.Second):
				t.Fatal("login never reached credential storage")
			}

			for _, path := range []string{"/api/login", "/api/register"} {
				response, err := request(http.MethodPost, path, "192.0.2.2")
				require.NoError(t, err, "overload must reject promptly without waiting for the occupied password slot")
				assert.Equal(t, http.StatusServiceUnavailable, response.StatusCode)
				assert.Equal(t, "2", response.Header.Get("Retry-After"))
				assert.Empty(t, response.Cookies(), "rejected authentication cannot create a session")
				_ = response.Body.Close()
			}
			assert.Equal(t, int32(1), repository.credentialCalls.Load())
			assert.Zero(t, repository.userWrites.Load())
			response, err := request(http.MethodGet, "/api/health", "192.0.2.3")
			require.NoError(t, err)
			assert.Equal(t, http.StatusOK, response.StatusCode, "ordinary requests remain available while password work is blocked")
			_ = response.Body.Close()
			select {
			case <-first:
				t.Fatal("the original credential lookup should still be blocked")
			default:
			}

			close(repository.release)
			completed := <-first
			require.NoError(t, completed.err)
			wantStatus := http.StatusOK
			if failure {
				wantStatus = http.StatusInternalServerError
			}
			assert.Equal(t, wantStatus, completed.response.StatusCode)
			_ = completed.response.Body.Close()
			response, err = request(http.MethodPost, "/api/login", "192.0.2.2")
			require.NoError(t, err)
			assert.Equal(t, http.StatusOK, response.StatusCode, "completion must release the slot even after repository failure")
			assert.NotEmpty(t, response.Cookies())
			_ = response.Body.Close()
			wantSessions := int32(2)
			if failure {
				wantSessions = 1
			}
			assert.Equal(t, wantSessions, repository.sessionWrites.Load())
		})
	}
}
