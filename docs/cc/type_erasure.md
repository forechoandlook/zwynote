

```cpp
std::any a;             // a is empty
a = 4.3;                // a has value 4.3 of type double
a = 42;                 // a has value 42 of type int
a = std::string{"hi"};  // a has value "hi" of type std::string
if (a.type() == typeid(std::string)) {
  std::string s = std::any_cast<std::string>(a);
  UseString(s);
} else if (a.type() == typeid(int)) {
  UseInt(std::any_cast<int>(a));
}
```

std::any可以用来表示任何可拷贝构造的单值类型,对类型的数据进行了抽象。除了对类型的数据进行抽象外，也可以对类型的行为进行抽象，例如std::function可以用来表示所有的可被调用的对象：普通函数、成员函数、函数对象、lambda表达式。

具体实现是:

```cpp
class Any {
  // Holds either pointer to a heap object or the contained object itself.
  union Storage {
    constexpr Storage() : ptr{nullptr} {}

    // Prevent trivial copies of this type, buffer might hold a non-POD.
    Storage(const Storage&) = delete;
    Storage& operator=(const Storage&) = delete;

    void* ptr;
    std::aligned_storage_t<sizeof(ptr), alignof(void*)> buffer;
  };
  Storage storage_;
};
```

这个方法的优点是：
1. 小对象优化：直接存储在 buffer 中（当对象大小不超过指针大小时）
2. 大对象存储：使用 ptr 指向堆上分配的内存


```cpp
class Any {
  union Storage { /*...*/ };

  // 用于存储类型信息
  const std::type_info* type_info_;

  // 类型擦除的关键：函数指针表
  struct VTable {
      void (*destroy)(Storage&);
      void (*copy)(Storage&, const Storage&);
      void (*move)(Storage&, Storage&);
      const std::type_info& (*type)();
  };
  const VTable* vtable_;

  Storage storage_;
};
```

3. 工作原理
- 当存储一个新值时：
  1. 根据值的大小选择存储策略
  2. 创建对应类型的 VTable
  3. 保存类型信息
  4. 在 storage_ 中存储数据

类型擦除的核心在于：

1. 灵活的存储策略（union）
2. 虚函数表（VTable）记录类型相关的操作
3. 运行时类型信息（type_info）