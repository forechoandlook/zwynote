
## 交叉编译 

```bash
GOOS=linux   GOARCH=amd64 go build -o myapp-linux
GOOS=windows GOARCH=386   go build -o myapp.exe
GOOS=linux   GOARCH=arm64 go build -o myapp-arm64
GOOS=darwin  GOARCH=amd64 go build -o myapp-macos-intel   # Intel
GOOS=darwin  GOARCH=arm64 go build -o myapp-macos-silicon  # Apple M系列

go tool dist list
```
通过 `-ldflags` 注入编译时的版本号（如 `-X main.version=1.0.0`）。

使用 `upx `工具减小文件体积


Go 的编译过程主要分为以下几个阶段：

1. 词法分析 & 语法分析 → 生成抽象语法树（AST）

2. 类型检查与语义分析 → 确保代码逻辑正确

3. 中间代码生成 → 转换为 SSA（静态单赋值）形式

4. 机器码生成 → 针对目标平台（GOOS/GOARCH）生成机器码

5. 链接 → 静态链接依赖，生成最终二进制文件

- `go build -x`: 查看编译过程的详细信息


## 工具

- `go tool nm`: 查看二进制文件的符号表
- `go tool objdump`: 查看二进制文件的汇编代码
- `go tool pprof`: 分析性能数据
- `go tool trace`: 分析程序的执行轨迹
- `go tool vet`: 检查代码的潜在问题
- `go tool asm`: 查看汇编代码
- `go tool compile`: 编译 Go 代码

### debug 

`go install github.com/go-delve/delve/cmd/dlv@latest`

```bash
# 启动调试（进入交互式界面）
dlv debug main.go

# 附加到正在运行的进程
dlv attach <PID>

# 设置断点
(dlv) break main.main
(dlv) break foo.go:10

# 执行下一步
(dlv) next

# 打印变量
(dlv) print x

# 查看协程
(dlv) goroutines

# 继续运行
(dlv) continue
```

运行时分析工具 

```bash
go tool pprof http://localhost:6060/debug/pprof/heap  # 分析内存
go tool pprof http://localhost:6060/debug/pprof/profile  # 分析 CPU
```
```bash
# 协程调度跟踪
go run main.go --trace=trace.out
go tool trace trace.out
```

单元测试 
```bash
dlv test .  # 调试当前包的所有测试
dlv test ./pkg -- -test.run TestFoo  # 调试特定测试
```




```log
WORK=/var/folders/td/2br0c4zs5mx8w86qrk1tss340000gn/T/go-build331156467
mkdir -p $WORK/b001/
cat >/var/folders/td/2br0c4zs5mx8w86qrk1tss340000gn/T/go-build331156467/b001/importcfg.link << 'EOF' # internal
packagefile command-line-arguments=/Users/wangyangzuo/Library/Caches/go-build/cb/cb8a360cbf4de6a7ce5468ee572b19c7caa67ce3d0186cfba601b99bd65335fe-d
packagefile runtime=/Users/wangyangzuo/Library/Caches/go-build/e0/e0d0243fe544bb80026ffda6e63c5ebf56cb7b1722fad8f9da590a71b4674ba8-d
packagefile internal/abi=/Users/wangyangzuo/Library/Caches/go-build/f9/f9cc50b4663366c39cac8d6ed0b9fb1474a55b99912b0ed32a15697684b7805e-d
packagefile internal/bytealg=/Users/wangyangzuo/Library/Caches/go-build/8a/8aec2bb16289441b9cf9361d1fc832a3a49b4bd0f6cfbf2bf08802278acf8faa-d
packagefile internal/byteorder=/Users/wangyangzuo/Library/Caches/go-build/1a/1ac33c4603f1e493062b433693964a3550dbbecc60d671540190355ca9f7a4f9-d
packagefile internal/chacha8rand=/Users/wangyangzuo/Library/Caches/go-build/9c/9cd3c6719cb309954ff0602b358659c2740ad3b598ea764050740e38ce264213-d
packagefile internal/coverage/rtcov=/Users/wangyangzuo/Library/Caches/go-build/ca/ca4c0c159cf4c2dd6bf468e59773e782021a22a1577eacb9b0324e158a97167d-d
packagefile internal/cpu=/Users/wangyangzuo/Library/Caches/go-build/e5/e55bfed6575cfe848b3a3d9b22a73d239c3438dff027d8a4b74bcc714a00b66a-d
packagefile internal/goarch=/Users/wangyangzuo/Library/Caches/go-build/70/702ec68599075ff341b90e7bc123b941c8f7a792fbe6965c17850ce746020e7a-d
packagefile internal/godebugs=/Users/wangyangzuo/Library/Caches/go-build/9f/9fdaf50a907b00f746c9f1d12961188702f70088b4b0c970372601ee342e32df-d
packagefile internal/goexperiment=/Users/wangyangzuo/Library/Caches/go-build/cb/cb497f3c33d6c0d90db143972ae18f28bf6a949424e4ecf731b68517757d1e15-d
packagefile internal/goos=/Users/wangyangzuo/Library/Caches/go-build/26/266ed5ca3258affee2809c07308e67c9297f93b2e4f8ee63ca3e411894112f23-d
packagefile internal/profilerecord=/Users/wangyangzuo/Library/Caches/go-build/78/7887ac0a837c3e42c476c0ea96e9485d2afd0d372b6052f1b8c6f394c21e132b-d
packagefile internal/runtime/atomic=/Users/wangyangzuo/Library/Caches/go-build/2f/2f4c17c1099989057a31bfd49a2e96a9d7513aaf92025b811224f560f9fb9c78-d
packagefile internal/runtime/exithook=/Users/wangyangzuo/Library/Caches/go-build/e6/e663a8586909b36a2e2c89df9a44e3a78f9e9a165b9ff7a4d5e6cdc9d6c2c79e-d
packagefile internal/runtime/maps=/Users/wangyangzuo/Library/Caches/go-build/79/79805c503c65eeb5d0a8f0b530b94f4784ef9b5233de40b04352c131a54e9352-d
packagefile internal/runtime/math=/Users/wangyangzuo/Library/Caches/go-build/13/1321b2588497f634cefdbf36b23f230d7e048d2ea08d55d4f5d48f1fa1890b70-d
packagefile internal/runtime/sys=/Users/wangyangzuo/Library/Caches/go-build/3d/3d3dda4daf7aeabc38d6045915a51482e7a1968d3cab5103557f54b9be204844-d
packagefile internal/stringslite=/Users/wangyangzuo/Library/Caches/go-build/f7/f74cb148242ddc363a10889010abed82f9222bfa0d73ba348ac1286c04457232-d
packagefile internal/asan=/Users/wangyangzuo/Library/Caches/go-build/39/3964d694cfa049b1e3b408007f65ee2d997be5c3d95cedcad4bf99765a46dc00-d
packagefile internal/msan=/Users/wangyangzuo/Library/Caches/go-build/74/74a05562bbf4218932c5a50b0d20ef90190b9a8db50456cbf3262d67f9bf6f5a-d
packagefile internal/race=/Users/wangyangzuo/Library/Caches/go-build/ee/ee5d3bed6b478dea4dc2a9e838f2f273d9d10574e320a53897df5d22bb7bf934-d
modinfo "0w\xaf\f\x92t\b\x02A\xe1\xc1\a\xe6\xd6\x18\xe6path\tcommand-line-arguments\nbuild\t-buildmode=exe\nbuild\t-compiler=gc\nbuild\t-ldflags=\"-s -w\"\nbuild\tCGO_ENABLED=1\nbuild\tCGO_CFLAGS=\nbuild\tCGO_CPPFLAGS=\nbuild\tCGO_CXXFLAGS=\nbuild\tCGO_LDFLAGS=\nbuild\tGOARCH=arm64\nbuild\tGOOS=darwin\nbuild\tGOARM64=v8.0\n\xf92C1\x86\x18 r\x00\x82B\x10A\x16\xd8\xf2"
EOF
mkdir -p $WORK/b001/exe/
cd .
GOROOT='/opt/homebrew/Cellar/go/1.24.2/libexec' /opt/homebrew/Cellar/go/1.24.2/libexec/pkg/tool/darwin_arm64/link -o $WORK/b001/exe/a.out -importcfg $WORK/b001/importcfg.link -buildmode=pie -buildid=1WwIR0-q08jpRVU0OLja/kD6mBAn_dyo6TeqULTnB/V2uspWB9SmSMbGnbLoUg/1WwIR0-q08jpRVU0OLja -s -w -extld=cc /Users/wangyangzuo/Library/Caches/go-build/cb/cb8a360cbf4de6a7ce5468ee572b19c7caa67ce3d0186cfba601b99bd65335fe-d
/opt/homebrew/Cellar/go/1.24.2/libexec/pkg/tool/darwin_arm64/buildid -w $WORK/b001/exe/a.out # internal
mv $WORK/b001/exe/a.out myapp
rm -rf $WORK/b001/
```