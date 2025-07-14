


build:
	jupyter-book build .

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
