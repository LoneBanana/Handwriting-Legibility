// Everything that depends on the exact Tavus "interactions" message format over
// the Daily data channel lives here, so it can be adjusted in one place.
//
// - After each answer's `done` event the page sends ONE echo interaction with
//   that event's `echo_text` (done: true): the replica speaks it verbatim, and
//   its <emotion value="…"/> tags steer the replica's (Phoenix) face. Splitting
//   an answer across several echo messages is not documented by Tavus.
// - In echo mode Tavus doesn't listen to the learner; questions always go
//   through the app's own /messages stream.
// - "Stop" sends an interrupt interaction.
// - The replica reports when it starts/stops speaking, which paces captions.

/** Anything that can send a Daily app message (a DailyCall in practice). */
export type AppMessageSender = { sendAppMessage: (data: unknown, to?: string) => unknown }

export type TavusInteraction = {
  message_type: 'conversation'
  event_type: string
  conversation_id: string
  properties?: Record<string, unknown>
}

/** Daily's "send to every participant" recipient. */
export const ALL_PARTICIPANTS = '*'

export function echoInteraction(conversationId: string, echoText: string): TavusInteraction {
  return {
    message_type: 'conversation',
    event_type: 'conversation.echo',
    conversation_id: conversationId,
    properties: { modality: 'text', text: echoText, done: true },
  }
}

export function interruptInteraction(conversationId: string): TavusInteraction {
  return { message_type: 'conversation', event_type: 'conversation.interrupt', conversation_id: conversationId }
}

/** Makes the replica say `echoText` (with its emotion tags). */
export function sendEcho(call: AppMessageSender, conversationId: string, echoText: string): void {
  call.sendAppMessage(echoInteraction(conversationId, echoText), ALL_PARTICIPANTS)
}

/** Asks the replica to stop talking. */
export function sendInterrupt(call: AppMessageSender, conversationId: string): void {
  call.sendAppMessage(interruptInteraction(conversationId), ALL_PARTICIPANTS)
}

/**
 * Reads the replica's "started/stopped speaking" broadcasts from a Daily
 * app-message payload: true / false, or null for anything else.
 */
export function replicaSpeakingFrom(data: unknown): boolean | null {
  if (!data || typeof data !== 'object') return null
  const eventType = (data as { event_type?: unknown }).event_type
  if (eventType === 'conversation.replica.started_speaking') return true
  if (eventType === 'conversation.replica.stopped_speaking') return false
  return null
}
