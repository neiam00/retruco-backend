package main

import (
	"fmt"
	"strconv"
)

func passwordDuty() (int, error) {
	n, err := strconv.Atoi(env("PASSWORD_WORK_DUTY_PERCENT", "20"))
	if err != nil || n < 1 || n > 100 {
		return 0, fmt.Errorf("PASSWORD_WORK_DUTY_PERCENT must be an integer from 1 to 100")
	}
	return n, nil
}
