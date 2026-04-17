import { Component, type ReactNode } from 'react'
import { BlockMath } from 'react-katex'
import 'katex/dist/katex.min.css'

interface Props {
  latex: string
}

interface State {
  hasError: boolean
}

/** LaTeX 렌더링 실패 시 plain text로 fallback하는 Error Boundary */
export class FormulaBlock extends Component<Props, State> {
  constructor(props: Props) {
    super(props)
    this.state = { hasError: false }
  }

  static getDerivedStateFromError(): State {
    return { hasError: true }
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return <code className="font-mono text-xs text-gray-700">{this.props.latex}</code>
    }
    return <BlockMath math={this.props.latex} />
  }
}
