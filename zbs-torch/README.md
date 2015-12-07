# ZBS-torch

원문: <https://github.com/soumith/zbs-torch/>

[지비에스-토치(ZBS-torch)](https://github.com/soumith/zbs-torch)는 토치7과 함께 동작하기 위한 제로브레인(ZeroBrane) 스튜디오의 한 포크(fork:기트허브에서 원본 프로젝트의 소스 코드 복사본을 내 원격 저장소로 내려받는 행위 또는 그 행위의 결과물)입니다.

제로브레인 스튜디오의 개요는 다음 문서를 참고하십시오. [README-zbs](https://github.com/soumith/zbs-torch/blob/master/README-zbs.md)

* 루아로 써있습니다, 그래서 쉽게 환경에 맞춰 바꿀 수 있습니다(customizable).
* 작고, 이식 가능하고, 플랫폼 독립적입니다 (윈도우즈, 맥 OS X, 리눅스).
* 함수, 키워드, 그리고 커스텀 API들을 위한 자동 완성.
* 로컬(내 컴퓨터) 또는 원격 실행으로 작은 코드를 바로 시험하기 위한 상호작용적 콘솔.
* 통합된 디버거, 로컬에서 그리고 [원격 디버깅](http://studio.zerobrane.com/doc-remote-debugging)
([루아 5.1](http://studio.zerobrane.com/doc-lua-debugging)을 위한,
[루아 5.2](http://studio.zerobrane.com/doc-lua52-debugging)을 위한,
[루아 5.3](http://studio.zerobrane.com/doc-lua53-debugging)을 위한,
[LuaJIT](http://studio.zerobrane.com/doc-luajit-debugging)을 위한,
그리고 [다른 루아 엔진들](http://studio.zerobrane.com/documentation#debugging)을 위한).
* [라이브 코딩](http://studio.zerobrane.com/documentation#live_coding)
 ([루아](http://notebook.kulchenko.com/zerobrane/live-coding-in-lua-bret-victor-style)로,
[LÖVE](http://notebook.kulchenko.com/zerobrane/live-coding-with-love)로,
[Gideros](http://notebook.kulchenko.com/zerobrane/gideros-live-coding-with-zerobrane-studio-ide)로,
[Moai](http://notebook.kulchenko.com/zerobrane/live-coding-with-moai-and-zerobrane-studio)로,
[Corona SDK](http://notebook.kulchenko.com/zerobrane/debugging-and-live-coding-with-corona-sdk-applications-and-zerobrane-studio)로,
GSL-shell로, 그리고 다른 엔진들로).
* 함수 개요 설명(outline).
* `Go To File`, `Go To Symbol`, 그리고 `Insert Library Function`를 이용한 퍼지 탐색.
* 현재 기능성을 확장하기 위한 몇 가지 방법들:
  - spec들 (`spec/`): 파일 문법(syntax), 어휘 분석기(lexer), 그리고 키워드들을 위한 설계 명세서;
  - api들 (`api/`): [코드 완성과 툴팁들](http://studio.zerobrane.com/doc-api-auto-complete)을 위한 설명들;
  - 인터프리터들 (`interpreters/`): 디버깅과 런타임 프로젝트 환경을 위한 요소들;
  - 패키지들 (`packages/`): 추가적 기능성을 제공하는 [플러그인들](http://studio.zerobrane.com/doc-plugin);
  - 구성 (`cfg/`): 스타일, 색 테마, 그리고 다른 속성들을 위한 설정들;
  - 번역들 (`cfg/i18n/`): [번역들](http://studio.zerobrane.com/doc-translation) 다른 언어로 쓰인 메뉴들과 메시지들;
  - 도구들 (`tools/`): 추가적 도구들.

## 설치
=======
* 토치 설치

* 루아락스로 몹디버그(mobdebug) 설치

```bash
$ luarocks install mobdebug
```

```bash
$ git clone https://github.com/soumith/zbs-torch.git
$ cd zbs-torch
$ ./zbstudio.sh
```

## 사용

토치 파일을 디버그 하기 위해,

* zbs를 zbs-torch 디렉토리에서 다음 명령어로 실행합니다.

```bash
$ ./zbstudio.sh
```
* "Project -> Start Debugger Server"로 디버거 서버를 시작합니다.

* "Project -> Lua Interpreter -> Torch-7"로 인터프리터를 Torch-7으로 바꿉니다.

* 우리가 디버깅하고 있는 파일의 맨 처음에 다음 줄을 추가합니다.

```lua
require('mobdebug').start()
```
예를 들어, 이 파일을
```lua
require 'image'
print('Wheres Waldo?')
a=image.rotate(image.lena(), 1.0)
image.display(a)
print('OK Bye')
```
아래 처럼 바꿉니다.
```lua
require('mobdebug').start()
require 'image'
print('Wheres Waldo?')
a=image.rotate(image.lena(), 1.0)
image.display(a)
print('OK Bye')
```

* "Project -> Run" 메뉴에서 그 파일을 실행합니다.
* 우선 그 파일의 첫 줄에서 디버거 멈춤(stop)으로 시작합니다, 그런 다음 우리는 중단점(breakpoint) 설정, 계속(continue), 그리고 단계적 실행 등의 일을 할 수 있습니다.

## 원작자

### 제로브레인 스튜디오 그리고 몹디버그 (ZeroBrane Studio and MobDebug)

  **ZeroBrane LLC:** Paul Kulchenko (paul@kulchenko.com)
## 라이센스

[라이센스](LICENSE)를 보십시오.
