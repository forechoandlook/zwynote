package database

import (
	"testing"
)

func TestInitDB(t *testing.T) {
	db := InitDB()
	sqlDB, err := db.DB()
	if err != nil {
		t.Errorf("数据库连接获取失败: %v", err)
	}

	// 测试数据库连接
	err = sqlDB.Ping()
	if err != nil {
		t.Errorf("数据库连接测试失败: %v", err)
	}
}
