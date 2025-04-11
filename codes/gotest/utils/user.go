package utils

import (
	"zwy/test/model"

	"gorm.io/gorm"
)

// CreateUser 创建用户
func CreateUser(db *gorm.DB, name string, email string, age int, isActive bool) (*model.User, error) {
	user := model.User{
		Name:     name,
		Email:    email,
		Age:      age,
		IsActive: isActive,
	}
	result := db.Create(&user)
	if result.Error != nil {
		return nil, result.Error
	}
	return &user, nil
}

// GetUserByID 通过ID查询用户
func GetUserByID(db *gorm.DB, id uint) (*model.User, error) {
	var user model.User
	result := db.First(&user, id)
	if result.Error != nil {
		return nil, result.Error
	}
	return &user, nil
}

// UpdateUser 更新用户信息
func UpdateUser(db *gorm.DB, user *model.User, updates map[string]interface{}) error {
	result := db.Model(user).Updates(updates)
	return result.Error
}

// GetAllUsers 获取所有用户
func GetAllUsers(db *gorm.DB) ([]model.User, error) {
	var users []model.User
	result := db.Find(&users)
	if result.Error != nil {
		return nil, result.Error
	}
	return users, nil
}

// DeleteUser 删除用户
func DeleteUser(db *gorm.DB, user *model.User) error {
	return db.Delete(user).Error
}
