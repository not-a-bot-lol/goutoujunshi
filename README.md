# 狗头军师 · Windows 0.1 测试版

这是狗头军师 Windows 端的 **0.1 测试版**。它在微信或 QQ 的聊天窗口旁边读屏、核对原文，并生成回复草稿。发送始终由你自己完成，程序不会代点发送。

这个版本还在测试。微信 Windows 4.x 和当前 QQ（`QQ.exe` / `QQNT.exe`）可以试，但不保证每个版本的窗口布局都能认对。识图可能把说话人认反，填入前请核对会话和草稿。

## 能做什么

- 读取当前可见的微信或 QQ 聊天窗口。两个都开着时，点一下要读的那个窗口，采集会跟着前台走。
- 核对原文和说话人之后，生成候选回复。可以复制，也可以填进输入框。
- 自己的消息：微信按绿色气泡认，QQ 按蓝色或青色气泡认。
- 设置保存在这台电脑的当前用户下。模型、关系和识图方式在 `%APPDATA%\goutoujunshi-jev-chat\config.json`，API 密钥在当前用户的环境变量里，不写进仓库。

## 源码运行

需要 Windows 10 1903 或 Windows 11，以及 Python 3.10–3.12。

```powershell
cd integrations\jev_windows
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python main.py
```

打开微信或 QQ 的主聊天窗口后再开采集。窗口缩在托盘里时读不到画面。

## 打包

在 `integrations\jev_windows` 里运行 `build.bat`。把生成的整个 `dist\goutoujunshi-jev-chat-windows` 文件夹一起拷走，再双击其中的 `goutoujunshi-jev-chat-windows.exe`。不要只拷贝 exe。

## 使用时注意

- 先点「核对原文并分析」，确认「我 / 对方」和文字，再看候选。
- 「填入草稿」只粘贴到输入框。如果点到表情栏，改用复制后自己粘贴。
- 换过气泡颜色时，说话人可能认错，在核对页里改。
- 对方明确要求停止联系时，这一轮不生成候选。
