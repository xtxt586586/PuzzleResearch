export type FrameScheduler = (
  callback: () => void,
) => number

export type FrameCanceller = (frameId: number) => void

export function createStreamTextBatcher(
  onFlush: (text: string) => void,
  scheduleFrame: FrameScheduler = (callback) =>
    window.requestAnimationFrame(callback),
  cancelFrame: FrameCanceller = (frameId) =>
    window.cancelAnimationFrame(frameId),
) {
  let chunks: string[] = []
  let frameId: number | null = null

  const flush = () => {
    if (frameId !== null) {
      cancelFrame(frameId)
      frameId = null
    }

    if (chunks.length === 0) {
      return
    }

    const text = chunks.join("")
    chunks = []
    onFlush(text)
  }

  return {
    append(text: string) {
      if (!text) {
        return
      }

      chunks.push(text)
      if (frameId === null) {
        frameId = scheduleFrame(() => {
          frameId = null
          flush()
        })
      }
    },
    flush,
    dispose() {
      if (frameId !== null) {
        cancelFrame(frameId)
        frameId = null
      }
      chunks = []
    },
  }
}
