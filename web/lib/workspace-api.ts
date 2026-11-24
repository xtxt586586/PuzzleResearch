import { getApiBaseUrl } from "./api"


export type WorkspaceFilesResponse = {
  thread_id: string
  count: number
  files: string[]
}

export type WorkspaceUploadResponse = {
  thread_id: string
  path: string
  bytes: number
}


async function readError(response: Response): Promise<string> {
  try {
    const payload = (await response.json()) as { detail?: unknown }
    if (typeof payload.detail === "string") return payload.detail
  } catch {
    // Keep the status as the fallback when the response is not JSON.
  }
  return `工作区请求失败：${response.status}`
}


export async function listWorkspaceFiles(
  threadId: string,
): Promise<WorkspaceFilesResponse> {
  const response = await fetch(
    `${getApiBaseUrl()}/api/sessions/${encodeURIComponent(threadId)}/workspace/files`,
    { cache: "no-store", headers: { Accept: "application/json" } },
  )
  if (!response.ok) throw new Error(await readError(response))
  return (await response.json()) as WorkspaceFilesResponse
}


export async function downloadWorkspaceFile(
  threadId: string,
  path: string,
): Promise<Blob> {
  const encodedPath = path.split("/").map(encodeURIComponent).join("/")
  const response = await fetch(
    `${getApiBaseUrl()}/api/sessions/${encodeURIComponent(threadId)}/workspace/files/${encodedPath}`,
  )
  if (!response.ok) throw new Error(await readError(response))
  return response.blob()
}


export async function uploadWorkspaceFile(
  threadId: string,
  file: File,
): Promise<WorkspaceUploadResponse> {
  const form = new FormData()
  form.append("file", file)
  const response = await fetch(
    `${getApiBaseUrl()}/api/sessions/${encodeURIComponent(threadId)}/workspace/files/upload`,
    { method: "POST", body: form, headers: { Accept: "application/json" } },
  )
  if (!response.ok) throw new Error(await readError(response))
  return (await response.json()) as WorkspaceUploadResponse
}


export async function deleteWorkspaceFile(
  threadId: string,
  path: string,
): Promise<void> {
  const encodedPath = path.split("/").map(encodeURIComponent).join("/")
  const response = await fetch(
    `${getApiBaseUrl()}/api/sessions/${encodeURIComponent(threadId)}/workspace/files/${encodedPath}`,
    { method: "DELETE", headers: { Accept: "application/json" } },
  )
  if (!response.ok) throw new Error(await readError(response))
}
