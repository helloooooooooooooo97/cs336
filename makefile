# 环境构建
venv:
	@# 创建 Python 虚拟环境（如果不存在）
	@if [ ! -d "venv" ]; then \
		python3 -m venv venv || python -m venv venv; \
		echo "虚拟环境已创建"; \
	else \
		echo "虚拟环境已存在"; \
	fi

# 依赖安装
install: venv
	@# 安装依赖（如有 requirements.txt 文件）
	@if [ -f "requirements.txt" ]; then \
		. venv/bin/activate && pip install -r requirements.txt; \
		echo "依赖已安装"; \
	else \
		echo "未找到 requirements.txt，跳过依赖安装"; \
	fi

# 环境激活
activate:
	@# 激活虚拟环境，提示用户根据操作系统选择命令
	@echo "请根据您的操作系统使用以下命令激活虚拟环境："
	@echo "  - macOS/Linux: source venv/bin/activate"
	@echo "  - Windows: venv\\Scripts\\activate"

# 编译jupyterbook
build:
	jupyter-book build .

# 打开jupyterbook网站
open:
	@# 判断操作系统并打开生成的HTML首页
	@if [ "$$(uname)" = "Darwin" ]; then \
		open _build/html/index.html; \
	elif [ "$$(uname)" = "Linux" ]; then \
		xdg-open _build/html/index.html; \
	elif [ "$$(uname | grep -i 'mingw')" != "" ]; then \
		start _build/html/index.html; \
	else \
		echo "请手动打开 _build/html/index.html"; \
	fi