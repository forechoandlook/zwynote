package utils

import (
	"testing"
	"zwy/test/database"
)

func TestUserCRUD(t *testing.T) {
	db := database.InitDB()

	// 测试创建用户
	t.Run("CreateUser", func(t *testing.T) {
		user, err := CreateUser(db, "测试用户", "test@example.com", 30, true)
		if err != nil {
			t.Errorf("创建用户失败: %v", err)
		}
		if user.ID == 0 {
			t.Error("用户ID不应该为0")
		}
		if user.Name != "测试用户" {
			t.Errorf("用户名不匹配，期望: 测试用户，实际: %s", user.Name)
		}
	})

	// 测试查询用户
	t.Run("GetUserByID", func(t *testing.T) {
		// 先创建一个用户
		user, _ := CreateUser(db, "查询测试", "query@example.com", 25, true)

		// 测试查询
		foundUser, err := GetUserByID(db, user.ID)
		if err != nil {
			t.Errorf("查询用户失败: %v", err)
		}
		if foundUser.Email != "query@example.com" {
			t.Errorf("用户邮箱不匹配，期望: query@example.com，实际: %s", foundUser.Email)
		}
	})

	// 测试更新用户
	t.Run("UpdateUser", func(t *testing.T) {
		// 先创建一个用户
		user, _ := CreateUser(db, "更新测试", "update@example.com", 28, true)

		// 更新用户信息
		updates := map[string]interface{}{
			"name": "更新后的名字",
			"age":  29,
		}
		err := UpdateUser(db, user, updates)
		if err != nil {
			t.Errorf("更新用户失败: %v", err)
		}

		// 验证更新结果
		updatedUser, _ := GetUserByID(db, user.ID)
		if updatedUser.Name != "更新后的名字" {
			t.Errorf("用户名更新失败，期望: 更新后的名字，实际: %s", updatedUser.Name)
		}
		if updatedUser.Age != 29 {
			t.Errorf("用户年龄更新失败，期望: 29，实际: %d", updatedUser.Age)
		}
	})

	// 测试获取所有用户
	t.Run("GetAllUsers", func(t *testing.T) {
		users, err := GetAllUsers(db)
		if err != nil {
			t.Errorf("获取所有用户失败: %v", err)
		}
		if len(users) == 0 {
			t.Error("用户列表不应该为空")
		}
	})

	// 测试删除用户
	t.Run("DeleteUser", func(t *testing.T) {
		// 先创建一个用户
		user, _ := CreateUser(db, "删除测试", "delete@example.com", 35, true)

		// 删除用户
		err := DeleteUser(db, user)
		if err != nil {
			t.Errorf("删除用户失败: %v", err)
		}

		// 验证用户已被删除
		_, err = GetUserByID(db, user.ID)
		if err == nil {
			t.Error("用户应该已被删除")
		}
	})

	// 清理测试数据
	db.Exec("DELETE FROM users")
}
