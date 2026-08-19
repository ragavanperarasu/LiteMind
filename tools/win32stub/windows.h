/* A minimal Win32 stub, only for syntax-checking the Windows code paths on a
   non-Windows host. It declares exactly what LiteMind uses and implements
   nothing. */
#pragma once
#include <cstddef>
#include <cstdint>

#define WINAPI
using HANDLE = void*;
using HMODULE = void*;
using PVOID = void*;
using LPVOID = void*;
using LPCVOID = const void*;
using SIZE_T = std::size_t;
using ULONG_PTR = std::uintptr_t;
using ULONG = unsigned long;
using DWORD = unsigned long;
using BOOL = int;
using LPSTR = char*;
using LPCWSTR = const wchar_t*;
using LPSECURITY_ATTRIBUTES = void*;
using FARPROC = int (WINAPI*)();
using LANGID = unsigned short;

union LARGE_INTEGER { long long QuadPart; };
struct SYSTEM_INFO { DWORD dwPageSize; };

#define INVALID_HANDLE_VALUE ((HANDLE)(long long)-1)
#define GENERIC_READ 0x80000000L
#define FILE_SHARE_READ 0x00000001
#define OPEN_EXISTING 3
#define FILE_ATTRIBUTE_NORMAL 0x80
#define PAGE_READONLY 0x02
#define FILE_MAP_READ 0x0004
#define FORMAT_MESSAGE_ALLOCATE_BUFFER 0x00000100
#define FORMAT_MESSAGE_FROM_SYSTEM 0x00001000
#define FORMAT_MESSAGE_IGNORE_INSERTS 0x00000200
#define LANG_NEUTRAL 0x00
#define SUBLANG_DEFAULT 0x01
#define MAKELANGID(p, s) ((((unsigned short)(s)) << 10) | (unsigned short)(p))
#define CP_UTF8 65001

extern "C" {
HANDLE CreateFileW(LPCWSTR, DWORD, DWORD, LPSECURITY_ATTRIBUTES, DWORD, DWORD, HANDLE);
BOOL GetFileSizeEx(HANDLE, LARGE_INTEGER*);
HANDLE CreateFileMappingW(HANDLE, LPSECURITY_ATTRIBUTES, DWORD, DWORD, DWORD, LPCWSTR);
LPVOID MapViewOfFile(HANDLE, DWORD, DWORD, DWORD, SIZE_T);
BOOL UnmapViewOfFile(LPCVOID);
BOOL CloseHandle(HANDLE);
DWORD GetLastError();
DWORD FormatMessageA(DWORD, LPCVOID, DWORD, DWORD, LPSTR, DWORD, void*);
void* LocalFree(void*);
void GetSystemInfo(SYSTEM_INFO*);
BOOL VirtualUnlock(LPVOID, SIZE_T);
HMODULE GetModuleHandleW(LPCWSTR);
FARPROC GetProcAddress(HMODULE, const char*);
HANDLE GetCurrentProcess();
BOOL SetConsoleOutputCP(unsigned int);
BOOL SetConsoleCP(unsigned int);
}
