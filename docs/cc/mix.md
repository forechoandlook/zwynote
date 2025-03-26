## mix
from : https://zhuanlan.zhihu.com/p/460825741

用途：将若干功能独立的类通过继承的方式实现模块复用的C++模板编程技巧

```c++
template<typename... Mixins>
class MixinClass : public Mixins... {
  public:
    MixinClass() :  Mixins...() {}
  // ...
};
```

`将模板参数作为派生类的基类`

```c++
template <typename... Mixins>
class Point : public Mixins... {
 public:
  double x, y;
  Point() : Mixins()..., x(0.0), y(0.0) {}
  Point(double x, double y) : Mixins()..., x(x), y(y) {}
};

class Label {
 public:
  std::string label;
  Label() : label("") {}
};

class Color {
 public:
  unsigned char red = 0, green = 0, blue = 0;
};

using MyPoint = Point<Label, Color>;
```

```c++
#include <iostream>
using namespace std;

struct Number
{
    typedef int value_type;
    int n;
    void set(int v) { n = v; }
    int get() const { return n; }
};

template <typename BASE, typename T = typename BASE::value_type>
struct Undoable
{
    typedef T value_type;
    BASE base;
    T before;
    void set(T v) { before = base.get(); base.set(v); }
    void undo() { base.set(before); }
    T get() const { return base.get(); }
};

template <typename BASE, typename T = typename BASE::value_type>
struct Redoable
{
    typedef T value_type;
    BASE base;
    T after;
    void set(T v) { after = v; base.set(v); }
    void redo() { base.set(after); }
    T get() const { return base.get(); }
};

typedef Redoable< Undoable<Number> > ReUndoableNumber;

int main()
{
    ReUndoableNumber mynum;
    mynum.set(42); mynum.set(84);
    cout << mynum.get() << '\n';  // 84
    mynum.undo();
    cout << mynum.get() << '\n';  // 42
    mynum.redo();
    cout << mynum.get() << '\n';  // back to 84
}
```

还可以通过这个实现可跟踪的异常,但是暂时不care了