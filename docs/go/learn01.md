## 汇编与go

```go
package main

func main() {
  _ = add(3,5)
}

func add(a, b int) int {
  return a+b
}
```

`go tool compile -S main.go` 


Go汇编指令格式是：操作码 + 源操作数 + 目标操作数的形式。

Go 汇编会在指令后加上 B , W , L 或 Q , 分别表示操作数的大小为1个，2个，4个或8个字节。

```go
MOVB $1, DI      // 1 byte
MOVW $0x10, BX   // 2 bytes
MOVL $1, DX      // 4 bytes
MOVQ $-10, AX    // 8 bytes   
```


