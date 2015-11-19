[![Join the chat at https://gitter.im/torch/torch7](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/torch/torch7?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/torch/torch7.svg)](https://travis-ci.org/torch/torch7)

## 도움이 필요하세요? ##
기터 채트(Gitter Chat)는 토치에 친숙한 개발자들과 사용자들을 위한 것입니다. 
토치 설치에 대한 질문이나 도움 요청은 [구글 그룹스 메일링 목록](https://groups.google.com/forum/#!forum/torch7)에 해주십시오.
우리의 대화창에 많은 양의 텍스트를 포스팅하거나 설치에 대해 묻는 것은 피해주십시오.
대신 그런 이슈들을 우리의 구글 그룹스 메일링 목록에서 다뤄주신다면 우리에게 큰 도움이 될 것입니다. :)

<a name="torch.reference.dok"/>
# 토치 패키지 참조 매뉴얼 #

__토치__는 다차원 텐서(tensor)들을 위한 자료 구조와 수학 연산이 정의되어 있는 Torch7의 주된 패키지입니다. 
추가적으로, 토치는 파일 접근, 임의 타입 객체의 직렬화, 그리고 다른 많은 유용한 지원 프로그램들을 제공합니다.

<a name="torch.overview.dok"/>
## 토치 패키지들 ##

  * 텐서 라이브러리
    * [텐서](https://docs.google.com/document/d/1QDtM8pdduWeUw0UK4zd7KgTuC6mFEbFCfVd3TF7Vj_Q/edit?usp=drive_web)는 다차원 수치 배열을 제공하는 모든 강력한 텐서 객체를 정의합니다. 그 텐서 객체에는 타입 템플리팅(templating) 기능이 있습니다.
    * 텐서 객체 타입들을 위해 정의된 [수학 연산들](https://docs.google.com/document/d/1js0VjoZ4HzixMVQvx7Lxw75ORCi-kRFoNqNrqFcUuNQ/edit?usp=drive_web).
    * [스토리지](https://docs.google.com/document/d/1Cl1ELxlAp66YjLJC83Vea4lecBeUEUr-r5ducBEhbqc/edit)는 텐서 객체를 위한 그 기저의 스토리지를 제어하는 간단한 스토리지 인터페이스를 정의합니다.
  * 파일 입/출력 인터페이스 라이브러리
    * [파일](https://docs.google.com/document/d/1KRatUyIfiXwFkNT3IN5kD5XIMJ5FPiaiJDQiijHGqB8/edit)은 파일 연산을 위한 추상 인터페이스입니다.
    * [디스크 파일](https://docs.google.com/document/d/12fM1sHbboQRWjUxVSjGGW6c7B_H8Vl7erqCYITQKnjs/edit)은 디스크에 저장되는 파일에 대한 연산들을 정의합니다.
    * [메모리 파일](https://docs.google.com/document/d/1XYBNVofo1FjY08fTDRLyC_KySKvz8KwRBLFluat6IPQ/edit)은 램(RAM)에 저장되는 연산들을 정의합니다.
    * [파이프 파일](https://docs.google.com/document/d/1SQQ6dq7t_eg35vQCYjqapSIKEKgQyEXqfcac5MDib38/edit)은 파이프화(piped) 명령들을 위한 연산들을 정의합니다.
    * [고수준 파일 연산들](https://docs.google.com/document/d/1vNdoIP_NGRwXhCfO2Z9bP8zu5MPVn3h66bwXGa2ZxYY/edit)은 더 높은 수준의 직렬화 함수들을 정의합니다.
  * 유용한 지원 프로그램들
    * [타이머](https://docs.google.com/document/d/1vOXoHQ5gQ8jRiJI4gRE8yk76GXdwpQriA0T3_EfoDLQ/edit)는 시간 측정을 위한 기능을 제공합니다.
    * [테스터](https://docs.google.com/document/d/1Oxa8KQ9hWKtCWFUvISprKcVV0OKIDbAagbxOu33pZd8/edit)는 일반적인 테스터 프레임워크입니다.
    * [CmdLine](https://docs.google.com/document/d/1c8vDU75d4CbVcXv-BuEHTLwE1mzcBbb-uoWdXqwu5Xc/edit)은 커맨드 라인 인자 구문 분석(parsing) 유틸리티입니다.
    * [랜덤](https://docs.google.com/document/d/1Tyzd3-UsKJxgtJj8RTMZw_cBstG0BbF3i3EqcN0bUvc/edit)은 다양한 분포를 가진 난수 발생기 패키지를 정의합니다.
    * 마지막으로 토치 텐서 타입과 클래스 상속을 쉽게 다루기 위한 유용한 [유틸리티](https://docs.google.com/document/d/1ELjQkdcqaWe0IIR7AbxFMLZmWCgbfDTFHUl8A-28GgY/edit) 함수들이 제공됩니다.

<a name="torch.links.dok"/>
## 유용한 링크들 ##

  * [유용한 자료 모음](https://github.com/torch/torch7/wiki/Cheatsheet)
  * [토치 블로그](http://torch.ch/blog/)
  * [토치 발표 자료](https://github.com/soumith/cvpr2015/blob/master/cvpr-torch.pdf)

