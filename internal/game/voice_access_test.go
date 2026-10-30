package game_test

import (
	"fmt"
	"io"
	"log/slog"
	"testing"

	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
	"github.com/neiam00/retruco-backend/rules"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVoiceAccessTracksCommittedMembershipAndRoomClosure(t *testing.T) {
	for _, spectatorsCanSpeak := range []bool{false, true} {
		t.Run(fmt.Sprintf("spectators_can_speak=%t", spectatorsCanSpeak), func(t *testing.T) {
			repository := &memoryRepository{payloads: make(map[string][]byte)}
			host := game.New(repository, slog.New(slog.NewTextHandler(io.Discard, nil)))
			owner := store.User{ID: "owner", Name: "Owner"}
			guest := store.User{ID: "guest", Name: "Guest"}
			code, err := host.Create(t.Context(), owner, rules.Config{Players: 4, Target: 12}, true,
				game.RoomOptions{SpectatorsCanSpeak: spectatorsCanSpeak})
			require.NoError(t, err)
			disconnect := host.ConnectPlayer(code, owner.ID, "owner-tab")
			t.Cleanup(disconnect)
			canSpeak, err := host.VoiceAccess(code, owner.ID)
			require.NoError(t, err)
			assert.True(t, canSpeak)
			canSpeak, err = host.VoiceAccess(code, guest.ID)
			require.NoError(t, err)
			assert.Equal(t, spectatorsCanSpeak, canSpeak, "knowing a private code grants the existing spectator voice policy")
			canSpeak, changes, unsubscribe, err := host.SubscribeVoiceAccess(code, guest.ID)
			require.NoError(t, err)
			t.Cleanup(unsubscribe)
			assert.Equal(t, spectatorsCanSpeak, canSpeak)
			quiet := func() {
				t.Helper()
				select {
				case <-changes:
					t.Fatal("unrelated or failed change woke voice authorization")
				default:
				}
			}
			changed := func() {
				t.Helper()
				select {
				case <-changes:
				default:
					t.Fatal("committed access change did not notify")
				}
			}
			require.NoError(t, host.Handle(t.Context(), code, owner, game.Input{ID: "chat", Chat: "hello"}))
			host.SetUpdating(true)
			host.SetUpdating(false)
			quiet()

			repository.fail = true
			require.Error(t, host.Join(t.Context(), code, guest, nil))
			quiet()
			canSpeak, err = host.VoiceAccess(code, guest.ID)
			require.NoError(t, err)
			assert.Equal(t, spectatorsCanSpeak, canSpeak, "failed storage cannot grant speaking permission")
			repository.fail = false
			require.NoError(t, host.Join(t.Context(), code, guest, nil))
			changed()
			canSpeak, err = host.VoiceAccess(code, guest.ID)
			require.NoError(t, err)
			assert.True(t, canSpeak)
			require.NoError(t, host.Handle(t.Context(), code, guest, game.Input{ID: "guest-leave", Command: rules.Command{Type: "leave"}}))
			changed()
			canSpeak, err = host.VoiceAccess(code, guest.ID)
			require.NoError(t, err)
			assert.Equal(t, spectatorsCanSpeak, canSpeak, "a departed player becomes a spectator")

			require.NoError(t, host.Handle(t.Context(), code, owner, game.Input{ID: "owner-leave", Command: rules.Command{Type: "leave"}}))
			changed()
			_, err = host.VoiceAccess(code, owner.ID)
			assert.Error(t, err, "removed rooms must revoke voice access even for the creator")
		})
	}
}
