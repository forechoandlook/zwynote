## 智能指针

```cpp
#include <iostream>
#include <cstring>
#include <atomic>

// ------------------- UniquePtr 实现 -------------------
template <typename T>
class UniquePtr {
private:
    T* ptr;
public:
    explicit UniquePtr(T* p = nullptr) : ptr(p) {}
    ~UniquePtr() { delete ptr; }

    UniquePtr(const UniquePtr&) = delete;
    UniquePtr& operator=(const UniquePtr&) = delete;

    UniquePtr(UniquePtr&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }
    UniquePtr& operator=(UniquePtr&& other) noexcept {
        if (this != &other) {
            delete ptr;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    T* operator->() const { return ptr; }
    T& operator*() const { return *ptr; }

    T* get() const { return ptr; }
};

// ------------------- SharedPtr 实现 -------------------
template <typename T>
class SharedPtr {
private:
    T* ptr;
    std::atomic<int>* count;
public:
    explicit SharedPtr(T* p = nullptr) : ptr(p), count(new std::atomic<int>(p ? 1 : 0)) {}

    ~SharedPtr() {
        if (--(*count) == 0) {
            delete ptr;
            delete count;
        }
    }

    SharedPtr(const SharedPtr& other) : ptr(other.ptr), count(other.count) {
        ++(*count);
    }

    SharedPtr& operator=(const SharedPtr& other) {
        if (this != &other) {
            if (--(*count) == 0) {
                delete ptr;
                delete count;
            }
            ptr = other.ptr;
            count = other.count;
            ++(*count);
        }
        return *this;
    }

    T* operator->() const { return ptr; }
    T& operator*() const { return *ptr; }

    T* get() const { return ptr; }
};

// ------------------- MyString 实现 -------------------
class MyString {
private:
    char* data;
    size_t length;
public:
    MyString(const char* str = "") {
        length = std::strlen(str);
        data = new char[length + 1];
        std::strcpy(data, str);
    }

    MyString(const MyString& other) {
        length = other.length;
        data = new char[length + 1];
        std::strcpy(data, other.data);
    }

    MyString& operator=(const MyString& other) {
        if (this != &other) {
            delete[] data;
            length = other.length;
            data = new char[length + 1];
            std::strcpy(data, other.data);
        }
        return *this;
    }

    MyString(MyString&& other) noexcept : data(other.data), length(other.length) {
        other.data = nullptr;
        other.length = 0;
    }

    MyString& operator=(MyString&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            length = other.length;
            other.data = nullptr;
            other.length = 0;
        }
        return *this;
    }

    ~MyString() { delete[] data; }

    const char* c_str() const { return data; }
};

// ------------------- 测试代码 -------------------
int main() {
    UniquePtr<int> uPtr(new int(10));
    std::cout << "UniquePtr: " << *uPtr << std::endl;

    SharedPtr<int> sPtr1(new int(20));
    SharedPtr<int> sPtr2 = sPtr1;
    std::cout << "SharedPtr: " << *sPtr1 << std::endl;

    MyString str1("Hello, World!");
    MyString str2 = str1;
    std::cout << "MyString: " << str2.c_str() << std::endl;

    return 0;
}

```


```cpp
template <typename T>
constexpr typename std::remove_reference<T>::type&& move(T&& arg) noexcept {
    return static_cast<typename std::remove_reference<T>::type&&>(arg);
}
```

std::move 无条件 转换为右值。
std::forward 有条件 转换为右值，仅用于 完美转发（配合模板）。

```cpp
template <typename T>
void foo(T&& arg) {
    T new_value = std::forward<T>(arg); // 保持左值/右值属性
}
```

## 万能引用和右值引用

绑定右值（没有名字的临时对象） 的引用，允许 移动语义。
1. 专门用于右值（int&& 只能绑定 10，不能绑定 int a;）。
2. 通常用于移动构造、移动赋值，避免拷贝，提高性能。
```cpp
void foo(int&& x) { // 右值引用
    std::cout << x << std::endl;
}

int main() {
    int a = 10;
    foo(10);   // ✅ 右值 OK
    foo(a);    // ❌ 左值不能传给 int&&
}
```
当 T&& 出现在函数模板中，并且 T 是模板参数，它变成 万能引用，可以接受左值和右值。
1. T&& 在模板参数中时是万能引用，可以接受左值和右值。
2. 需要 std::forward<T>(arg) 来保持原有的左值/右值特性。
3. 用于泛型编程，使函数可以接受各种类型的参数。
```cpp
template <typename T>
void wrapper(T&& arg) { // T&& 是万能引用
    foo(std::forward<T>(arg)); // 保持左值或右值的特性
}

int main() {
    int a = 10;
    wrapper(a);   // 传入左值，T = int&
    wrapper(20);  // 传入右值，T = int
}
```


```cpp
template <typename T, typename... Args>
void emplace_back(Args&&... args) {
    data.push_back(T(std::forward<Args>(args)...));
}
```
直接在容器中构造对象，避免了拷贝和移动操作。

`vec.emplace_back(1, 2.5, "Hello");  // 直接在容器中构造，避免临时对象`

在高性能应用中，使用 emplace_back 可以显著减少不必要的拷贝，提高代码的效率


```cpp
#include <iostream>

template <typename T>
class SharedPtr {
private:
    T* ptr;
    int* ref_count;

public:
    explicit SharedPtr(T* p = nullptr) : ptr(p), ref_count(new int(1)) {}

    ~SharedPtr() {
        if (--(*ref_count) == 0) {
            delete ptr;
            delete ref_count;
        }
    }

    SharedPtr(const SharedPtr& other) noexcept : ptr(other.ptr), ref_count(other.ref_count) {
        ++(*ref_count);
    }

    SharedPtr& operator=(const SharedPtr& other) noexcept {
        if (this != &other) {
            if (--(*ref_count) == 0) {
                delete ptr;
                delete ref_count;
            }
            ptr = other.ptr;
            ref_count = other.ref_count;
            ++(*ref_count);
        }
        return *this;
    }

    T* get() const { return ptr; }
    T* operator->() const { return ptr; }
    T& operator*() const { return *ptr; }

    int use_count() const { return *ref_count; }
};

struct Test {
    void show() { std::cout << "SharedPtr works!\n"; }
};

int main() {
    SharedPtr<Test> p1(new Test());
    SharedPtr<Test> p2 = p1;
    p2->show();
    std::cout << "Reference count: " << p1.use_count() << std::endl;
    return 0;
}
```

```cpp
#include <iostream>

template <typename T>
class SharedPtr; // 前向声明

template <typename T>
class WeakPtr {
private:
    T* ptr;
    int* ref_count;
    int* weak_count; // 额外的引用计数，用来管理 weak_ptr 的数量

public:
    WeakPtr() : ptr(nullptr), ref_count(nullptr), weak_count(nullptr) {}

    WeakPtr(const SharedPtr<T>& shared) : ptr(shared.ptr), ref_count(shared.ref_count), weak_count(shared.weak_count) {
        if (weak_count) {
            ++(*weak_count); // 增加 weak_ptr 数量
        }
    }

    ~WeakPtr() {
        if (weak_count && --(*weak_count) == 0) {
            delete weak_count;
        }
    }

    // lock() 方法：返回一个 shared_ptr，如果资源已经被销毁，返回空的 shared_ptr
    SharedPtr<T> lock() const;

    // 获取对象是否有效
    bool expired() const {
        return *ref_count == 0;
    }
};

template <typename T>
class SharedPtr {
private:
    T* ptr;
    int* ref_count; // 引用计数
    int* weak_count; // weak_ptr 引用计数

public:
    explicit SharedPtr(T* p = nullptr) : ptr(p), ref_count(new int(1)), weak_count(new int(0)) {}

    ~SharedPtr() {
        if (--(*ref_count) == 0) {
            delete ptr;
            delete ref_count;
            if (*weak_count == 0) {
                delete weak_count;
            }
        }
    }

    SharedPtr(const SharedPtr& other) noexcept : ptr(other.ptr), ref_count(other.ref_count), weak_count(other.weak_count) {
        ++(*ref_count);
    }

    SharedPtr& operator=(const SharedPtr& other) noexcept {
        if (this != &other) {
            if (--(*ref_count) == 0) {
                delete ptr;
                delete ref_count;
                if (*weak_count == 0) {
                    delete weak_count;
                }
            }
            ptr = other.ptr;
            ref_count = other.ref_count;
            weak_count = other.weak_count;
            ++(*ref_count);
        }
        return *this;
    }

    SharedPtr(SharedPtr&& other) noexcept : ptr(other.ptr), ref_count(other.ref_count), weak_count(other.weak_count) {
        other.ptr = nullptr;
        other.ref_count = nullptr;
        other.weak_count = nullptr;
    }

    SharedPtr& operator=(SharedPtr&& other) noexcept {
        if (this != &other) {
            if (--(*ref_count) == 0) {
                delete ptr;
                delete ref_count;
                if (*weak_count == 0) {
                    delete weak_count;
                }
            }
            ptr = other.ptr;
            ref_count = other.ref_count;
            weak_count = other.weak_count;
            other.ptr = nullptr;
            other.ref_count = nullptr;
            other.weak_count = nullptr;
        }
        return *this;
    }

    friend class WeakPtr<T>;

    // 返回原始指针
    T* get() const { return ptr; }
    int use_count() const { return *ref_count; }
};

// weak_ptr::lock 实现
template <typename T>
SharedPtr<T> WeakPtr<T>::lock() const {
    if (*ref_count > 0) {
        return SharedPtr<T>(*this); // 创建一个新的 shared_ptr
    } else {
        return SharedPtr<T>(); // 返回一个空的 shared_ptr
    }
}

struct Test {
    void show() { std::cout << "Test class works!\n"; }
};

int main() {
    SharedPtr<Test> sp1(new Test());
    {
        WeakPtr<Test> wp1(sp1); // wp1 观察 sp1
        if (!wp1.expired()) {
            SharedPtr<Test> sp2 = wp1.lock(); // 从 weak_ptr 获取 shared_ptr
            sp2->show(); // 输出：Test class works!
        }
    } // wp1 离开作用域，弱引用不再引用资源

    if (sp1.use_count() == 0) {
        std::cout << "sp1 has no references left." << std::endl;
    }
    return 0;
}
```