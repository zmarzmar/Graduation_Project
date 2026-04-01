declare module 'react-katex' {
  import { FC } from 'react'
  export const InlineMath: FC<{ math: string }>
  export const BlockMath: FC<{ math: string }>
}
