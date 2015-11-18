<a name="paths.wellknowndirs.dok"></a>
## 디렉토리 경로들 ##

이 변수들은 경로들을 가리킵니다. 그 경로는 다양한 토치 요소들이 설치되어 있고, 
또한 `$HOME` 같은 다른 흔한 환경 변수들을 가리키는 경로를 말합니다.
그 값들은 바꾸지 말기를 권합니다!

<a name="paths.install_prefix"></a>
### paths.install_prefix ###

기본 토치 설치 디렉토리.

<a name="paths.install_bin"></a>
### paths.install_bin ###

실행할 수 있는 프로그램들이 있는 그 디렉토리의 이름.
윈도우즈에서, 이 디렉토리는 또한 
동적 로드할 수 있는 라이브러리 (`.dll`)를 포함합니다.

<a name="paths.install_man"></a>
### paths.install_man ###

유닉스 스타일 매뉴얼 페이지들이 있는 디렉토리의 이름.

<a name="paths.install_lib"></a>
### paths.install_lib ###

객체 코드 라이브러리들이 있는 디렉토리 이름.
유닉스에서, 이 디렉토리는 또한 동적 로드할 수 있는 
라이브러리들(`.so` or `.dylib`)을 포함합니다.

<a name="paths.install_share"></a>
### paths.install_share ###

프로세서 독립 데이터 파일들이 있는 디렉토리 이름.
루아 코드나 다른 텍스트 파일들이 여기에 해당합니다.

<a name="paths.install_include"></a>
### paths.install_include ###

다양한 토치 라이브러리들을 위한 
인클루드 파일이 있는 디렉토리의 이름.

<a name="paths.install_hlp"></a>
### paths.install_hlp ###

토치 도움말 파일들이 있는 디렉토리 이름.

<a name="paths.install_html"></a>
### paths.install_html ###

HTML 버전의 토치 도움말 파일들이 있는 디렉토리의 이름.
이 파일들은 당신이 CMake 옵션 `HTML_DOC`을 활설화 할 때
만들어집니다.

<a name="paths.install_cmake"></a>
### paths.install_cmake ###

외부 토치 모듈들에서 사용되는 CMake 파일들이 있는
디렉토리의 이름.

<a name="paths.install_lua_path"></a>
### paths.install_lua_path ###

루아 패키지들이 있는 디렉토리의 이름.
이 디렉토리는 변수 `package.path`를 빌드하는 데 사용됩니다.

<a name="paths.install_lua_cpath"></a>
### paths.install_lua_cpath ###

루아가 로드할 수 있는 바이너리 모듈들이 있는 디렉토리의 이름.
이 디렉토리는 변수 `package.cpath`를 빌드하는 데 사용됩니다.

<a name="paths.home"></a>
### paths.home ###

현재 사용자의 홈 디렉토리.

