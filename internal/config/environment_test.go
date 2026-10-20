package config_test

import (
	"os"
	"testing"

	"github.com/neiam00/retruco-backend/internal/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDevelopmentEnvLoading(t *testing.T) {
	for _, mode := range []string{"development", "production", ""} {
		t.Run(mode, func(t *testing.T) {
			t.Chdir(t.TempDir())
			t.Setenv("APP_ENV", mode)
			t.Setenv("RETRUCO_TEST_EXISTING", "from-shell")
			t.Setenv("RETRUCO_TEST_EMPTY", "")
			t.Setenv("RETRUCO_TEST_LOADED", "")
			require.NoError(t, os.Unsetenv("RETRUCO_TEST_LOADED"))
			require.NoError(t, os.WriteFile(".env", []byte(
				"RETRUCO_TEST_LOADED='from file'\n"+
					"RETRUCO_TEST_EXISTING=from-file\n"+
					"RETRUCO_TEST_EMPTY=from-file\n",
			), 0o600))
			require.NoError(t, os.WriteFile(".env.example", []byte("RETRUCO_TEST_LOADED=example\n"), 0o600))

			require.NoError(t, config.LoadDevelopmentEnv())

			value, loaded := os.LookupEnv("RETRUCO_TEST_LOADED")
			if mode == "development" {
				assert.True(t, loaded)
				assert.Equal(t, "from file", value)
			} else {
				assert.False(t, loaded)
			}
			assert.Equal(t, "from-shell", os.Getenv("RETRUCO_TEST_EXISTING"))
			assert.Empty(t, os.Getenv("RETRUCO_TEST_EMPTY"))
		})
	}
}

func TestMissingEnvDoesNotLoadExample(t *testing.T) {
	t.Chdir(t.TempDir())
	t.Setenv("APP_ENV", "development")
	t.Setenv("RETRUCO_TEST_LOADED", "")
	require.NoError(t, os.Unsetenv("RETRUCO_TEST_LOADED"))
	require.NoError(t, os.WriteFile(".env.example", []byte("RETRUCO_TEST_LOADED=example\n"), 0o600))

	require.NoError(t, config.LoadDevelopmentEnv())
	_, loaded := os.LookupEnv("RETRUCO_TEST_LOADED")
	assert.False(t, loaded)
}

func TestMalformedEnvFailsOnlyInDevelopmentWithoutExposingContents(t *testing.T) {
	t.Chdir(t.TempDir())
	require.NoError(t, os.WriteFile(".env", []byte("SECRET='private-unclosed-value\n"), 0o600))
	t.Setenv("APP_ENV", "production")
	require.NoError(t, config.LoadDevelopmentEnv())

	t.Setenv("APP_ENV", "development")
	err := config.LoadDevelopmentEnv()
	require.Error(t, err)
	assert.Contains(t, err.Error(), ".env")
	assert.NotContains(t, err.Error(), "private-unclosed-value")
}
