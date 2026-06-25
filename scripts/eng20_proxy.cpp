#include <stdio.h>
#include <stdarg.h>

typedef void *HMODULE;
typedef void *FARPROC;
typedef unsigned long DWORD;
typedef unsigned short WORD;
typedef unsigned char BYTE;

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

EXTERN_C __declspec(dllimport) HMODULE __stdcall LoadLibraryA(const char *);
EXTERN_C __declspec(dllimport) FARPROC __stdcall GetProcAddress(HMODULE, const char *);
EXTERN_C __declspec(dllimport) DWORD __stdcall GetLastError(void);
EXTERN_C __declspec(dllimport) DWORD __stdcall GetTickCount(void);
EXTERN_C __declspec(dllimport) int __stdcall IsBadReadPtr(const void *, unsigned int);

typedef struct CDictInfo {
    unsigned char bytes[0x190];
} CDictInfo;

typedef int (__stdcall *FnInit)(char *, CDictInfo);
typedef void (__stdcall *FnTerm)();
typedef void *(__stdcall *FnRecogLineEngStr)(unsigned char *, short, short, void *);
typedef void *(__stdcall *FnRecogImgEngStr)(unsigned char *, short, short, void *, int (__cdecl *)(unsigned int), int (__cdecl *)(void));
typedef void (__stdcall *FnFreeRgnInfoEngStr)(void *);

static HMODULE g_real = NULL;
static FnInit g_init = NULL;
static FnTerm g_term = NULL;
static FnRecogLineEngStr g_recog_line_engstr = NULL;
static FnRecogImgEngStr g_recog_img_engstr = NULL;
static FnFreeRgnInfoEngStr g_free_engstr = NULL;

static void log_line(const char *fmt, ...)
{
    FILE *fp = fopen("eng20_hook.log", "ab");
    if (!fp) {
        return;
    }
    fprintf(fp, "tick=%lu ", GetTickCount());
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp, fmt, ap);
    va_end(ap);
    fprintf(fp, "\r\n");
    fclose(fp);
}

static int appendf(char *buf, int cap, int pos, const char *fmt, ...)
{
    if (pos >= cap) {
        return pos;
    }
    va_list ap;
    va_start(ap, fmt);
    int wrote = vsnprintf(buf + pos, cap - pos, fmt, ap);
    va_end(ap);
    if (wrote < 0) {
        return pos;
    }
    pos += wrote;
    if (pos >= cap) {
        pos = cap - 1;
        buf[pos] = '\0';
    }
    return pos;
}

static int readable(const void *ptr, unsigned int bytes)
{
    return ptr && !IsBadReadPtr(ptr, bytes);
}

static WORD read_u16(const void *base, int offset)
{
    const BYTE *ptr = (const BYTE *)base + offset;
    if (!readable(ptr, 2)) {
        return 0xffff;
    }
    return *(const WORD *)ptr;
}

static int read_i16(const void *base, int offset)
{
    return (short)read_u16(base, offset);
}

static int read_i32(const void *base, int offset)
{
    const BYTE *ptr = (const BYTE *)base + offset;
    if (!readable(ptr, 4)) {
        return 0x7fffffff;
    }
    return *(const int *)ptr;
}

static void *read_ptr(const void *base, int offset)
{
    const BYTE *ptr = (const BYTE *)base + offset;
    if (!readable(ptr, 4)) {
        return NULL;
    }
    return *(void * const *)ptr;
}

static void log_recblocks(const char *label, void *head)
{
    if (!head) {
        log_line("%s recblock=(null)", label);
        return;
    }
    void *block = head;
    int index = 0;
    while (block && index < 32) {
        int a = read_i16(block, 0);
        int b = read_i16(block, 2);
        int c = read_i16(block, 4);
        int d = read_i16(block, 6);
        void *next = read_ptr(block, 8);
        log_line(
            "%s recblock[%d] raw=(%d,%d,%d,%d) as_ltrb=(l=%d,t=%d,r=%d,b=%d) as_tblr=(t=%d,b=%d,l=%d,r=%d) next=%p",
            label,
            index,
            a,
            b,
            c,
            d,
            a,
            b,
            c,
            d,
            a,
            b,
            c,
            d,
            next);
        block = next;
        index++;
    }
    if (block) {
        log_line("%s recblock truncated_after=%d next=%p", label, index, block);
    }
}

static void log_engstr_result(const char *label, void *region_info)
{
    if (!region_info) {
        log_line("%s result=(null)", label);
        return;
    }

    int line_index = 0;
    void *region = region_info;
    int region_index = 0;
    while (region && region_index < 32 && line_index < 256) {
        void *line = read_ptr(region, 4);
        int local_line_index = 0;
        while (line && local_line_index < 128 && line_index < 256) {
            char ascii[512];
            char codes[2048];
            char boxes[4096];
            int ascii_pos = 0;
            int codes_pos = 0;
            int boxes_pos = 0;
            int char_count = 0;
            int group_index = 0;
            void *group = read_ptr(line, 4);
            ascii[0] = '\0';
            codes[0] = '\0';
            boxes[0] = '\0';

            while (group && group_index < 256 && char_count < 512) {
                if (group_index > 0) {
                    ascii_pos = appendf(ascii, sizeof(ascii), ascii_pos, "|");
                }
                void *chr = read_ptr(group, 4);
                int group_char_index = 0;
                while (chr && group_char_index < 512 && char_count < 512) {
                    WORD code = read_u16(chr, 2);
                    WORD score = read_u16(chr, 0x16);
                    int top = read_u16(chr, 0x2a);
                    int bottom = read_u16(chr, 0x2c);
                    int left = read_u16(chr, 0x2e);
                    int right = read_u16(chr, 0x30);

                    char visible = '.';
                    if (code >= 32 && code < 127) {
                        visible = (char)code;
                    }
                    ascii_pos = appendf(ascii, sizeof(ascii), ascii_pos, "%c", visible);
                    codes_pos = appendf(codes, sizeof(codes), codes_pos, "%s%u", char_count ? "," : "", code);
                    if (char_count < 80) {
                        boxes_pos = appendf(
                            boxes,
                            sizeof(boxes),
                            boxes_pos,
                            "%s%d:%d,%d,%d,%d/%u",
                            char_count ? ";" : "",
                            char_count,
                            left,
                            top,
                            right,
                            bottom,
                            score);
                    }

                    chr = read_ptr(chr, 0x38);
                    group_char_index++;
                    char_count++;
                }
                group = read_ptr(group, 0x18);
                group_index++;
            }

            log_line(
                "%s result line=%d region=%d raw_count=%d flags=%d groups=%d chars=%d ascii=\"%s\" codes=[%s] boxes=[%s]",
                label,
                line_index,
                region_index,
                read_i16(line, 0),
                read_i32(line, 0x0c),
                group_index,
                char_count,
                ascii,
                codes,
                boxes);

            line = read_ptr(line, 0x10);
            local_line_index++;
            line_index++;
        }
        region = read_ptr(region, 0x10);
        region_index++;
    }

    if (region || line_index >= 256) {
        log_line("%s result truncated line_count=%d next_region=%p", label, line_index, region);
    }
}

static FARPROC load_proc(const char *name)
{
    if (!g_real) {
        g_real = LoadLibraryA("Eng20_real.dll");
        if (!g_real) {
            log_line("LoadLibrary Eng20_real.dll failed err=%lu", GetLastError());
            return NULL;
        }
    }
    FARPROC proc = GetProcAddress(g_real, name);
    if (!proc) {
        log_line("GetProcAddress failed name=%s err=%lu", name, GetLastError());
    }
    return proc;
}

static void ensure_loaded()
{
    if (!g_init) {
        g_init = (FnInit)load_proc("?HW_ENG20_Init@@YGHPADUCDictInfo@@@Z");
    }
    if (!g_term) {
        g_term = (FnTerm)load_proc("?HW_ENG20_Term@@YGXXZ");
    }
    if (!g_recog_line_engstr) {
        g_recog_line_engstr = (FnRecogLineEngStr)load_proc("?HW_ENG20_RECOGLINE_ENGSTR@@YGPAU_regioninfoeng@@PAEFFPAU_recblockeng@@@Z");
    }
    if (!g_recog_img_engstr) {
        g_recog_img_engstr = (FnRecogImgEngStr)load_proc("?HW_ENG20_RECOGIMG_ENGSTR@@YGPAU_regioninfoeng@@PAEFFPAU_recblockeng@@P6AHI@ZP6AHXZ@Z");
    }
    if (!g_free_engstr) {
        g_free_engstr = (FnFreeRgnInfoEngStr)load_proc("?HW_ENG20_FREERGNINFO_ENGSTR@@YGXPAU_regioninfoeng@@@Z");
    }
}

EXTERN_C int __stdcall Proxy_HW_ENG20_Init(char *path, CDictInfo dictInfo)
{
    ensure_loaded();
    log_line("CALL Init path=%s", path ? path : "(null)");
    if (!g_init) {
        return -1;
    }
    int rc = g_init(path, dictInfo);
    log_line("RET  Init rc=%d", rc);
    return rc;
}

EXTERN_C void __stdcall Proxy_HW_ENG20_Term()
{
    ensure_loaded();
    log_line("CALL Term");
    if (g_term) {
        g_term();
    }
    log_line("RET  Term");
}

EXTERN_C void *__stdcall Proxy_HW_ENG20_RECOGLINE_ENGSTR(unsigned char *bits, short width, short height, void *recblock)
{
    ensure_loaded();
    log_line("CALL RECOGLINE_ENGSTR bits=%p width=%d height=%d recblock=%p", bits, width, height, recblock);
    log_recblocks("CALL RECOGLINE_ENGSTR", recblock);
    if (!g_recog_line_engstr) {
        return NULL;
    }
    void *ret = g_recog_line_engstr(bits, width, height, recblock);
    log_line("RET  RECOGLINE_ENGSTR ret=%p", ret);
    log_engstr_result("RET  RECOGLINE_ENGSTR", ret);
    return ret;
}

EXTERN_C void *__stdcall Proxy_HW_ENG20_RECOGIMG_ENGSTR(
    unsigned char *bits,
    short width,
    short height,
    void *recblock,
    int (__cdecl *progress)(unsigned int),
    int (__cdecl *cancel)(void))
{
    ensure_loaded();
    log_line("CALL RECOGIMG_ENGSTR bits=%p width=%d height=%d recblock=%p progress=%p cancel=%p",
             bits, width, height, recblock, progress, cancel);
    log_recblocks("CALL RECOGIMG_ENGSTR", recblock);
    if (!g_recog_img_engstr) {
        return NULL;
    }
    void *ret = g_recog_img_engstr(bits, width, height, recblock, progress, cancel);
    log_line("RET  RECOGIMG_ENGSTR ret=%p", ret);
    log_engstr_result("RET  RECOGIMG_ENGSTR", ret);
    return ret;
}

EXTERN_C void __stdcall Proxy_HW_ENG20_FREERGNINFO_ENGSTR(void *regionInfo)
{
    ensure_loaded();
    log_line("CALL FREERGNINFO_ENGSTR ptr=%p", regionInfo);
    if (g_free_engstr) {
        g_free_engstr(regionInfo);
    }
    log_line("RET  FREERGNINFO_ENGSTR");
}
