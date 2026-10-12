#!/usr/bin/env bash
# daytrade-it PreToolUse 가드 — CLAUDE.md 의 "반드시" 규칙 셋을 결정론으로 막는다.
#  1) docs/research/frozen/ 의 기존 사전등록 파일 수정·삭제 (새 파일·amend 는 허용)
#  2) data/live_STOP 삭제·이동 (실매매 재개 = 사용자 결정)
#  3) 런타임 체크아웃(~/git/daytrade-it)에서 main 이외 브랜치로 checkout/switch
# exit 2 = 도구 호출 거부, stderr 가 Claude 에게 전달된다.
set -u
in=$(cat)
tool=$(jq -r '.tool_name // empty' <<<"$in")
deny() { echo "daytrade-it 가드: $1" >&2; exit 2; }

case "$tool" in
  Write|Edit|MultiEdit|NotebookEdit)
    f=$(jq -r '.tool_input.file_path // .tool_input.notebook_path // empty' <<<"$in")
    case "$f" in
      */docs/research/frozen/*)
        [ -e "$f" ] && deny "frozen 사전등록($f)은 고치지 않는다. 정정은 새 파일 <원본>-amendN.json 으로."
        ;;
    esac
    ;;
  Bash)
    cmd=$(jq -r '.tool_input.command // empty' <<<"$in")
    cwd=$(jq -r '.cwd // empty' <<<"$in")
    if grep -qE 'live_STOP' <<<"$cmd" && grep -qE '(^|[;&|[:space:]])(rm|mv|unlink|git[[:space:]]+rm)[[:space:]]' <<<"$cmd"; then
      deny "data/live_STOP 삭제·이동 = 실매매 재개. forward_gate promote=true + 사용자 명시 승인 뒤 사용자가 직접 한다."
    fi
    if grep -qE 'docs/research/frozen/' <<<"$cmd" && grep -qE '(^|[;&|[:space:]])(rm|mv|truncate|git[[:space:]]+(rm|mv|restore|checkout))[[:space:]]|sed[[:space:]]+(-[a-zA-Z]*i|--in-place)' <<<"$cmd"; then
      deny "frozen 사전등록 파일을 셸로 바꾸지 않는다. 정정은 새 amendN 파일로."
    fi
    rt="$HOME/git/daytrade-it"
    if grep -qE '\bgit\b.*\b(checkout|switch)\b' <<<"$cmd" \
       && { grep -qE "git[[:space:]]+-C[[:space:]]+(~|$HOME)/git/daytrade-it/?[[:space:]]" <<<"$cmd" \
            || { case "$cwd" in "$rt"|"$rt"/*) ! grep -qE 'git[[:space:]]+-C[[:space:]]' <<<"$cmd";; *) false;; esac; }; } \
       && ! grep -qE '\b(checkout|switch)[[:space:]]+main\b|[[:space:]]--[[:space:]]' <<<"$cmd"; then
      deny "~/git/daytrade-it 는 크론 런타임 체크아웃이다(main 고정). 작업은 git worktree add ../daytrade-it-<주제> -b <브랜치> origin/main 으로."
    fi
    ;;
esac
exit 0
