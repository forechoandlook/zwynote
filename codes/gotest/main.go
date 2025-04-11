package main

import (
	"fmt"
	"zwy/test/database"
	"zwy/test/utils"
)

func main() {
	db := database.InitDB()

	// 创建用户
	user, err := utils.CreateUser(db, "张三", "zhangsan@example.com", 25, true)
	if err != nil {
		fmt.Println("创建用户失败:", err)
		return
	}
	fmt.Println("创建用户成功，ID:", user.ID)

	// 查询用户
	foundUser, err := utils.GetUserByID(db, user.ID)
	if err != nil {
		fmt.Println("查询用户失败:", err)
		return
	}
	fmt.Printf("查询到的用户: %+v\n", foundUser)

	// 更新用户
	updates := map[string]interface{}{
		"name": "李四",
		"age":  26,
	}
	if updateErr := utils.UpdateUser(db, foundUser, updates); updateErr != nil {
		fmt.Println("更新用户失败:", err)
		return
	}
	fmt.Println("用户更新成功")

	// 查询所有用户
	users, err := utils.GetAllUsers(db)
	if err != nil {
		fmt.Println("查询所有用户失败:", err)
		return
	}
	fmt.Println("所有用户列表:")
	for _, u := range users {
		fmt.Printf("ID: %d, 姓名: %s, 邮箱: %s, 年龄: %d\n", u.ID, u.Name, u.Email, u.Age)
	}

	// 删除用户
	if err := utils.DeleteUser(db, foundUser); err != nil {
		fmt.Println("删除用户失败:", err)
		return
	}
	fmt.Println("用户删除成功")
}
