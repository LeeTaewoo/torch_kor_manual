<a name="paths.dirs.dok"></a>
## 디렉토리 함수들 ##

다음 함수들은 디렉토리 내용을 조사하거나
디렉토리들을 조작하는 데 사용될 수 있습니다.

<a name="paths.dir"></a>
### paths.dir(dname) ###

디렉토리 `dname` 안에 있는 파일들과 디렉토리들을 담은 테이블 하나를 리턴합니다.
만약 특정된 디렉토리가 존재하지 않으면, 이 함수는 `nil`을 리턴합니다.
리눅스에서, 이것은 `.`와 `..` 디렉토리도 포함합니다.

<a name="paths.files"></a>
### paths.files(dname [, include]) ###

디렉토리 `dname` 안에 있는 파일들과 디렉토리들에 대한 한 반복자(iteratoe)를 리턴합니다. 
리눅스에서, 이것은 `.`와 `..` 디렉토리도 포함합니다.
아래 보여지는 것처럼, 이것은 *__for__* 문에서 사용될 수 있습니다:

```lua
for f in paths.files(".") do
   print(f)
end
```

선택적 인자 `include`는 어떤 파일들이 포함될지는 결정하는 데 사용되는
함수 하나 또는 문자열 하나 입니다. 이 함수는 파일 이름을 인자로 받아,
만약 그 파일이 인클루드되면, true를 리턴해야 합니다.
만약 한 문자열이 제공되면, 다음 함수가 사용됩니다:

```lua
function(file) 
   return file:find(f) 
end
```

하위 폴더들의 파일들과 디렉토리들은 인클루드되지 않습니다.

<a name="paths.iterdirs"></a>
### paths.iterdirs(dname) ###

디렉토리 `dname` 안에 있는 디렉토리들에 대한 한 반복자(iteratoe)를 리턴합니다.
아래 보여지는 것처럼, 이것은 *__for__* 문에서 사용될 수 있습니다:

```lua
for dir in paths.iterdirs(".") do
   print(dir)
end
```
하위 폴더들의 디렉토리들, 그리고 `.`와 `..` 폴더들은 포함되지 않습니다.

<a name="paths.iterdirs"></a>
### paths.iterfiles(dname) ###

디렉토리 `dname` 안에 있는 (디렉토리들이 아닌) 파일들에 대한 한 반복자(iteratoe)를 리턴합니다.
아래 보여지는 것처럼, 이것은 *__for__* 문에서 사용될 수 있습니다:

```lua
for file in paths.iterfiles(".") do
   print(file)
end
```

하위 폴더들의 디렉토리들, 그리고 `.`와 `..` 폴더들은 포함되지 않습니다.

<a name="paths.mkdir"></a>
### paths.mkdir(s) ###

디렉토리 하나를 만듭니다.
성공 시 `true`를 리턴합니다.

<a name="paths.rmdir"></a>
### paths.rmdir(s) ###

빈 디렉토리 하나를 지웁니다.
성공 시 `true`를 리턴합니다.

<a name="paths.rmall"></a>
### paths.rmall(s, y) ###

재귀적으로 파일 또는 디렉토리 `s`와 그것의 내용들을 지웁니다.
인자 `y`는 반드시 문자열 `"yes"`여야 합니다.
성공 시 `true`를 리턴합니다.

