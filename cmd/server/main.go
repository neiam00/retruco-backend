package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/neiam00/retruco-backend/internal/api"
	"github.com/neiam00/retruco-backend/internal/config"
	"github.com/neiam00/retruco-backend/internal/game"
	"github.com/neiam00/retruco-backend/internal/store"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stdout, nil))
	if err := run(log); err != nil {
		log.Error("server stopped", "error", err)
		os.Exit(1)
	}
}

func run(log *slog.Logger) error {
	if err := config.LoadDevelopmentEnv(); err != nil {
		return err
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	config, err := pgxpool.ParseConfig(os.Getenv("DATABASE_URL"))
	if err != nil {
		return err
	}
	config.MaxConns, config.MinConns = 10, 1
	config.MaxConnLifetime = time.Hour
	config.MaxConnIdleTime = 5 * time.Minute
	pool, err := pgxpool.NewWithConfig(ctx, config)
	if err != nil {
		return err
	}
	defer pool.Close()
	startup, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()
	if err := pool.Ping(startup); err != nil {
		return err
	}
	repository := store.New(pool)
	host := game.New(repository, log)
	if err := host.Restore(startup); err != nil {
		return err
	}
	origins := strings.Split(env("ALLOWED_ORIGINS", "http://localhost:5173"), ",")
	for i := range origins {
		origins[i] = strings.TrimSpace(origins[i])
	}
	server, err := api.New(ctx, api.Config{
		Origins:       origins,
		SecureCookies: env("COOKIE_SECURE", "true") == "true",
		ProxySecret:   os.Getenv("API_PROXY_SECRET"),
		Turnstile: api.TurnstileConfig{
			SiteKey:     os.Getenv("TURNSTILE_SITE_KEY"),
			SecretKey:   os.Getenv("TURNSTILE_SECRET_KEY"),
			Development: os.Getenv("APP_ENV") == "development",
		},
		Voice: api.VoiceConfig{
			STUNURLs:   envList("VOICE_STUN_URLS"),
			TURNURLs:   envList("VOICE_TURN_URLS"),
			TURNSecret: os.Getenv("VOICE_TURN_SECRET"),
		},
	}, repository, host, log)
	if err != nil {
		return err
	}
	httpServer := &http.Server{
		Addr:              env("LISTEN_ADDR", "127.0.0.1:8080"),
		Handler:           server.Handler(),
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       15 * time.Second,
		IdleTimeout:       60 * time.Second,
		MaxHeaderBytes:    16384,
	}
	go host.Run(ctx)
	go maintainSessions(ctx, repository, log)
	serveErr := make(chan error, 1)
	go func() { serveErr <- httpServer.ListenAndServe() }()
	log.Info("server listening", "address", httpServer.Addr)
	select {
	case err := <-serveErr:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	case <-ctx.Done():
	}
	shutdown, done := context.WithTimeout(context.Background(), 10*time.Second)
	defer done()
	return httpServer.Shutdown(shutdown)
}

func env(name, fallback string) string {
	if value := os.Getenv(name); value != "" {
		return value
	}
	return fallback
}

func envList(name string) []string {
	var values []string
	for _, value := range strings.Split(os.Getenv(name), ",") {
		if value = strings.TrimSpace(value); value != "" {
			values = append(values, value)
		}
	}
	return values
}
