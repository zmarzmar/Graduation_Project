import { Component, Fragment, type ReactNode } from 'react'
import { InlineMath } from 'react-katex'
import 'katex/dist/katex.min.css'

interface SafeInlineMathProps {
  content: string
}

interface SafeInlineMathState {
  hasError: boolean
}

class SafeInlineMath extends Component<SafeInlineMathProps, SafeInlineMathState> {
  constructor(props: SafeInlineMathProps) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(): SafeInlineMathState {
    return { hasError: true }
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return <code className="font-mono text-xs text-gray-700">{this.props.content}</code>
    }
    return <InlineMath math={this.props.content} />
  }
}

const MATH_PATTERN = /(\$\$[\s\S]+?\$\$|\\\[[\s\S]+?\\\]|\$[^$\n]+?\$|\\\([\s\S]+?\\\))/g

function extractMathContent(raw: string): string {
  if (raw.startsWith('$$')) return raw.slice(2, -2).trim()
  if (raw.startsWith('\\[') || raw.startsWith('\\(')) return raw.slice(2, -2).trim()
  return raw.slice(1, -1).trim()
}

interface MathTextProps {
  text: string
  className?: string
}

/** 본문 텍스트 안의 인라인 수식($...$, \(...\), $$...$$, \[...\])을 KaTeX로 렌더링한다. */
export function MathText({ text, className }: MathTextProps) {
  const re = new RegExp(MATH_PATTERN.source, 'g')
  const parts: ReactNode[] = []
  let last = 0
  let m: RegExpExecArray | null

  while ((m = re.exec(text)) !== null) {
    if (m.index > last) parts.push(text.slice(last, m.index))
    parts.push(
      <SafeInlineMath key={`m-${m.index}`} content={extractMathContent(m[0])} />
    )
    last = m.index + m[0].length
  }
  if (last < text.length) parts.push(text.slice(last))

  if (parts.length === 0) {
    return className ? <span className={className}>{text}</span> : <Fragment>{text}</Fragment>
  }
  if (className) return <span className={className}>{parts}</span>
  return <Fragment>{parts}</Fragment>
}
