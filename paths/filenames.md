<a name="paths.filenames.dok"></a>
## 파일 이름 조작 ##

다음 함수들은 이식 가능한 방식으로
다양한 플랫폼에서 파일 이름을 조작하는 데
사용될 수 있습니다.

<a name="paths.filep"></a>
### paths.filep(path) ###

`path`가 존재하는 파일을 참조하는지 가리키는
한 불리언을 리턴합니다 .

<a name="paths.dirp"></a>
### paths.dirp(path) ###

`path`가 존재하는 디렉토리를 참조하는지를 가리키는
한 불리언을 리턴합니다.

<a name="paths.basename"></a>
### paths.basename(path,[suffix]) ###

`path`의 마지막 경로 요소를 리턴합니다
그리고 선택적으로 접미사 `suffix`를 벗겨냅니다.
이것은 잘 알려진 쉘 명령어 `"basename"`과 비슷합니다.

<a name="paths.dirname"></a>
### paths.dirname(path) ###

파일 `path`가 있는 디렉토리 이름을 리턴합니다.
이것은 잘 알려진 쉘 명령어 `"dirname"`과 비슷합니다.

<a name="paths.extname"></a>
### paths.extname(path) ###

`path`의 확장을 리턴합니다. 만약 아무것도 찾지 못하면,
nil을 리턴합니다.

<a name="paths.concat"></a>
### paths.concat([path1,....,pathn]) ###

상대적 파일 이름들을 이어붙입니다.

우선 이 함수는 `path1`의 현재 디렉토리에 상대적인 전체 파일이름을 계산합니다.
그런 다음, 이전 인자를 위해 리턴되는, 그 파일 이름에 상대적인, 인자들 `path2`에서 `pathn`까지의 전체 파일 이름들을 연속적으로 계산합니다. 끝으로, 그 마지막 결과가 리턴됩니다.

만약 이 함수를 인자 없이 호출하면, 현재 디렉토리의 전체 이름을 리턴합니다.

<a name="paths.cwd"></a>
### paths.cwd() ###

현재 작업 중인 디렉토리의 전체 경로를 리턴합니다.

<a name="paths.execdir"></a>
### paths.execdir() ###

현재 실행 가능한 루아 파일이 있는 
디렉토리의 이름을 리턴합니다.
모듈 `paths`가 처음 로드될 때, 
이 정보는 다양한 토치 요소들의 위치를 
가리키는 변수들을 재배치 하기 위해
사용됩니다.

<a name="paths.tmpname"></a>
### paths.tmpname() ###

한 임시 파일의 이름을 리턴합니다.
그 이름이 이 방식으로 얻어진 모든 임시 파일들은
루아가 종료될 때 함께 제거됩니다.

`os.tmpname()`보다는 이 함수를 주로 사용해야 합니다,
왜냐하면 이 함수는 루아가 종료될 때, 임시 파일들을 
확실히 제거하기 때문입니다.
게다가, 윈도우즈에서 `os.tmpname()`은 종종 
사용자가 쓸 수 있는 권한이 없는 파일 이름을 리턴합니다.
