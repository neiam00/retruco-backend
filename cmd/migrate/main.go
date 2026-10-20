package main

import (
	"context"
	"database/sql"
	"log"
	"os"

	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/neiam00/retruco-backend/internal/config"
	"github.com/pressly/goose/v3"
)

func main() {
	if err := config.LoadDevelopmentEnv(); err != nil {
		log.Fatal(err)
	}
	db, err := sql.Open("pgx", os.Getenv("DATABASE_URL"))
	if err != nil {
		log.Fatal(err)
	}
	defer func() {
		if err := db.Close(); err != nil {
			log.Printf("close database: %v", err)
		}
	}()
	if err := goose.SetDialect("postgres"); err != nil {
		log.Fatal(err)
	}
	command := "up"
	if len(os.Args) > 1 {
		command = os.Args[1]
	}
	if err := goose.RunContext(context.Background(), command, db, "migrations"); err != nil {
		log.Fatal(err)
	}
}
