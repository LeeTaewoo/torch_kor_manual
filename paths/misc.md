<a name="paths.misc.dok"></a>
## 그 밖의 것들 ##

<a name="paths.uname"></a>
### paths.uname() ###

운영 체제를 설명하는 최대 세 문자열들을 리턴합니다.
첫 번째 문자열은 시스템 이름입니다, 예를 들어, "Windows", "Linux", "Darwin", "FreeBSD"등이 있습니다.
두 번째 문자열은 이 컴퓨터의 네트워크 이름입니다.
세 번째 문자열은 프로세서 타입을 가리킵니다.

<a name="paths.is_win"></a>
### paths.is_win() ###

만약 운영 체제가 마이크로소프트 윈도우즈이면, true를 리턴합니다.

<a name="paths.is_mac"></a>
### paths.is_mac() ###

만약 운영 체제가 Mac OS X이면, true를 리턴합니다.

### paths.getregistryvalue(key,subkey,value) ###

윈도우즈 레지스트리 값에서 한 값을 문의합니다.
다른 시스템에서는 에러를 일으킬 수 있습니다.

### paths.findprogram(progname,...) ###

이름이 "progname"인 실행 가능한 프로그램을 찾습니다. 그리고 그것의 전체
경로를 리턴합니다. 만약 아무것도 찾이 못하면, 계속 
다음 인자들의 이름을 따서 명명된 프로그램들을 찾습니다,
그리고 처음으로 일치한 것의 전체 경로를 리턴합니다.
PATH 변수에 특정된 모든 디렉토리들이 검색됩니다.
윈도우즈에서, 이것은 또한 "App Path" 레지스트리 엔트리들을 검색합니다.

<a name="paths.findingfiles.dok"></a>
<a name="paths.thisfile"></a>
### paths.thisfile([arg]) ###

한 루아 파일 안에서 `paths.thisfile()`을 인자 없이 호출하는 것은
그것이 호출된 곳으로부터 그 파일의 전체 경로명을 리턴합니다.
상호작용적으로(th 콘솔에서) 호출될 때, 이 함수는 항상 `nil`을 리턴합니다.

`paths.thisfile(arg)`을 한 문자열 인자 `arg`와 함께 호출하는 것은
함수 `paths.thisfile`가 호출되는 곳으로부터 그 파일이 있는 
그 디렉토리에 상대적인  파일 `arg`의 전체 경로명을 리턴합니다.
예를 들어, 이것은 한 루아 스크립트로 같은 디렉토리에 위치한 
파일들의 위치를 찾을 때 유용합니다.

<a name="paths.dofile"></a>
### paths.dofile(filename) ###

이 함수는 표준 루아 함수 `dofile`과 비슷합니다.
그러나 한 디렉토리에 상대적으로 `filename`을 해석합니다
`paths.dofile`이 상호작용적으로(th 콘솔에서) 호출될 때, 그 디렉토리는 `paths.dofile`로의 호출 또는 현재 디렉토리로의 호출을 포함하는 파일을 포함합니다.
