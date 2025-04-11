package database

import (
    "log"
    "gorm.io/driver/sqlite"
    "gorm.io/gorm"
    "zwy/test/model"
)

func InitDB() *gorm.DB {
    db, err := gorm.Open(sqlite.Open("test.db"), &gorm.Config{})
    if err != nil {
        log.Fatal("Failed to connect database:", err)
    }

    // 自动迁移
    db.AutoMigrate(&model.User{})

    return db
}