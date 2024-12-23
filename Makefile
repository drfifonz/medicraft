

documentation:
	sphinx-build -b html .docs/ .docs/_build/html
	mkdir -p docs
	cp -r .docs/_build/html/* docs/

release:
	poetry version $(v)
	poetry build
	poetry publish

release-tag:
	git tag -a $(v) -m $(m)
	git push origin $(v)