package config

import (
	"errors"
	"os"

	"github.com/joho/godotenv"
)

// LoadDevelopmentEnv reads .env from the working directory only when the
// process explicitly opts into development. Existing environment values win.
func LoadDevelopmentEnv() error {
	if os.Getenv("APP_ENV") != "development" {
		return nil
	}
	if err := godotenv.Load(".env"); err != nil && !errors.Is(err, os.ErrNotExist) {
		// Parser errors may include file contents, so do not log the raw error.
		return errors.New("cannot load development .env: check its syntax and permissions")
	}
	return nil
}
