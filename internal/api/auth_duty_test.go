package api

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"testing/synctest"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPasswordDutyReservesTimeAndDoesNotChargePreHashFailures(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := &Server{config: Config{PasswordWorkDutyPercent: 20}, passwordSlots: make(chan struct{}, 1)}
		response := httptest.NewRecorder()
		release, ok := s.passwordAdmission(response)
		require.True(t, ok)
		// Slow storage before the hash must not create a cooldown.
		time.Sleep(5 * time.Second)
		release()
		release, ok = s.passwordAdmission(httptest.NewRecorder())
		require.True(t, ok)
		started := time.Now()
		time.Sleep(100 * time.Millisecond)
		s.passwordWorkCompleted(started)
		response = httptest.NewRecorder()
		_, ok = s.passwordAdmission(response)
		assert.False(t, ok, "handler retains the exclusive lease after hashing")
		assert.Equal(t, "2", response.Header().Get("Retry-After"))
		release()
		time.Sleep(399 * time.Millisecond)
		response = httptest.NewRecorder()
		_, ok = s.passwordAdmission(response)
		assert.False(t, ok)
		assert.Equal(t, http.StatusServiceUnavailable, response.Code)
		assert.Equal(t, "1", response.Header().Get("Retry-After"))
		time.Sleep(time.Millisecond)
		release, ok = s.passwordAdmission(httptest.NewRecorder())
		require.True(t, ok, "100ms work at 20 percent leaves 400ms for other work")
		release()
	})
}

func TestPasswordDutyRetryAfterRoundsUp(t *testing.T) {
	synctest.Test(t, func(t *testing.T) {
		s := &Server{config: Config{PasswordWorkDutyPercent: 20}, passwordSlots: make(chan struct{}, 1)}
		started := time.Now()
		time.Sleep(550 * time.Millisecond)
		s.passwordWorkCompleted(started)
		response := httptest.NewRecorder()
		_, ok := s.passwordAdmission(response)
		assert.False(t, ok)
		assert.Equal(t, "3", response.Header().Get("Retry-After"))
	})
}
