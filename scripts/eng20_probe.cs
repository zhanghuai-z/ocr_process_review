using System;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

internal static class Kernel32Native
{
    [DllImport("kernel32.dll", SetLastError = true)]
    internal static extern IntPtr GlobalAlloc(uint flags, UIntPtr bytes);

    [DllImport("kernel32.dll", SetLastError = true)]
    internal static extern IntPtr GlobalLock(IntPtr hMem);

    [DllImport("kernel32.dll", SetLastError = true)]
    internal static extern bool GlobalUnlock(IntPtr hMem);

    [DllImport("kernel32.dll", SetLastError = true)]
    internal static extern IntPtr GlobalFree(IntPtr hMem);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    internal static extern bool SetDllDirectory(string path);
}

[StructLayout(LayoutKind.Sequential, Size = 0x190)]
internal struct CDictInfo
{
}

internal static class Eng20Native
{
    [DllImport("Eng20.dll", CallingConvention = CallingConvention.StdCall, CharSet = CharSet.Ansi, EntryPoint = "?HW_ENG20_Init@@YGHPADUCDictInfo@@@Z")]
    internal static extern int Init(string path, CDictInfo dictInfo);

    [DllImport("Eng20.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "?HW_ENG20_Term@@YGXXZ")]
    internal static extern void Term();

    [DllImport("Eng20.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "?HW_ENG20_RECOGLINE@@YGPAU_regioninfo@@PAEFFPAU_recblock@@@Z")]
    internal static extern IntPtr RecogLine(IntPtr hBits, short width, short height, IntPtr recblock);

    [DllImport("Eng20.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "?HW_ENG20_RECOGIMG@@YGPAU_regioninfo@@PAEFFPAU_recblock@@P6AHI@ZP6AHXZ@Z")]
    internal static extern IntPtr RecogImg(IntPtr hBits, short width, short height, IntPtr recblock, IntPtr progressCallback, IntPtr cancelCallback);

    [DllImport("Eng20.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "?HW_ENG20_FREERGNINFO@@YGXPAU_regioninfo@@@Z")]
    internal static extern void FreeRgnInfo(IntPtr regionInfo);

    [DllImport("Eng20.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "?HW_ENG20_RECOGLINE_ENGSTR@@YGPAU_regioninfoeng@@PAEFFPAU_recblockeng@@@Z")]
    internal static extern IntPtr RecogLineEngStr(IntPtr hBits, short width, short height, IntPtr recblock);

    [DllImport("Eng20.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "?HW_ENG20_RECOGIMG_ENGSTR@@YGPAU_regioninfoeng@@PAEFFPAU_recblockeng@@P6AHI@ZP6AHXZ@Z")]
    internal static extern IntPtr RecogImgEngStr(IntPtr hBits, short width, short height, IntPtr recblock, IntPtr progressCallback, IntPtr cancelCallback);

    [DllImport("Eng20.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "?HW_ENG20_FREERGNINFO_ENGSTR@@YGXPAU_regioninfoeng@@@Z")]
    internal static extern void FreeRgnInfoEngStr(IntPtr regionInfo);

    [DllImport("Eng20.dll", CallingConvention = CallingConvention.StdCall, EntryPoint = "?Recognize@@YGHPAEHHPAG1PAF@Z")]
    internal static extern int Recognize(IntPtr hBits, int width, int height, ushort[] textA, ushort[] textB, short[] metrics);
}

internal static class Program
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int ProgressCallback(uint current);

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int CancelCallback();

    private static readonly byte[] Masks = new byte[8] { 128, 64, 32, 16, 8, 4, 2, 1 };
    private static readonly ProgressCallback Progress = OnProgress;
    private static readonly CancelCallback Cancel = OnCancel;

    private static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("usage: eng20_probe.exe <image> [output-json] [recblocks-tsv] [recogline|recogimg]");
            return 2;
        }

        string imagePath = Path.GetFullPath(args[0]);
        string outputPath = args.Length > 1 ? Path.GetFullPath(args[1]) : "";
        string baseDir = AppDomain.CurrentDomain.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar);
        string nativePath = baseDir.EndsWith("\\") ? baseDir : baseDir + "\\";
        string call = args.Length > 3 ? args[3] : "recogline";
        string imageFormat = args.Length > 4 ? args[4] : "packed";
        string recblockOrder = args.Length > 5 ? args[5] : "xyxy";

        Kernel32Native.SetDllDirectory(baseDir);

        int width;
        int height;
        byte[] packed = PackImage(imagePath, imageFormat, out width, out height);
        IntPtr hBits = AllocFixed(packed);
        IntPtr recblock = args.Length > 2 && File.Exists(args[2])
            ? BuildRecblockChain(args[2], recblockOrder)
            : BuildRecblock(width, height, recblockOrder);
        IntPtr regionInfo = IntPtr.Zero;
        bool engStrRegion = false;
        int init = -999999;
        int recognizeReturn = -999999;
        ushort[] recognizeTextA = new ushort[512];
        ushort[] recognizeTextB = new ushort[512];
        short[] recognizeMetrics = new short[512];
        string error = "";

        try
        {
            init = Eng20Native.Init(nativePath, new CDictInfo());
            if (call == "recognize")
            {
                recognizeReturn = Eng20Native.Recognize(hBits, width, height, recognizeTextA, recognizeTextB, recognizeMetrics);
            }
            else if (call == "recogline_engstr")
            {
                engStrRegion = true;
                regionInfo = Eng20Native.RecogLineEngStr(hBits, (short)width, (short)height, recblock);
            }
            else if (call == "recogimg_engstr")
            {
                engStrRegion = true;
                IntPtr progressPtr = Marshal.GetFunctionPointerForDelegate(Progress);
                IntPtr cancelPtr = Marshal.GetFunctionPointerForDelegate(Cancel);
                regionInfo = Eng20Native.RecogImgEngStr(hBits, (short)width, (short)height, recblock, progressPtr, cancelPtr);
            }
            else if (call == "recogimg")
            {
                IntPtr progressPtr = Marshal.GetFunctionPointerForDelegate(Progress);
                IntPtr cancelPtr = Marshal.GetFunctionPointerForDelegate(Cancel);
                regionInfo = Eng20Native.RecogImg(hBits, (short)width, (short)height, recblock, progressPtr, cancelPtr);
            }
            else
            {
                regionInfo = Eng20Native.RecogLine(hBits, (short)width, (short)height, recblock);
            }
        }
        catch (Exception ex)
        {
            error = ex.GetType().Name + ": " + ex.Message;
        }

        string json = BuildJson(
            imagePath,
            init,
            width,
            height,
            call,
            imageFormat,
            regionInfo,
            error,
            recognizeReturn,
            recognizeTextA,
            recognizeTextB,
            recognizeMetrics,
            engStrRegion);
        if (outputPath.Length > 0)
        {
            File.WriteAllText(outputPath, json, new UTF8Encoding(false));
        }
        else
        {
            Console.WriteLine(json);
        }

        if (regionInfo != IntPtr.Zero)
        {
            try
            {
                if (engStrRegion)
                {
                    Eng20Native.FreeRgnInfoEngStr(regionInfo);
                }
                else
                {
                    Eng20Native.FreeRgnInfo(regionInfo);
                }
            }
            catch
            {
            }
        }
        try
        {
            Eng20Native.Term();
        }
        catch
        {
        }
        if (recblock != IntPtr.Zero)
        {
            FreeRecblocks(recblock);
        }
        if (hBits != IntPtr.Zero)
        {
            Kernel32Native.GlobalFree(hBits);
        }
        return error.Length == 0 ? 0 : 1;
    }

    private static IntPtr AllocFixed(byte[] data)
    {
        IntPtr hMem = Kernel32Native.GlobalAlloc(64u, (UIntPtr)(ulong)data.Length);
        if (hMem == IntPtr.Zero)
        {
            throw new InvalidOperationException("GlobalAlloc failed");
        }
        IntPtr ptr = Kernel32Native.GlobalLock(hMem);
        if (ptr == IntPtr.Zero)
        {
            throw new InvalidOperationException("GlobalLock failed");
        }
        Marshal.Copy(data, 0, ptr, data.Length);
        Kernel32Native.GlobalUnlock(hMem);
        return hMem;
    }

    private static IntPtr BuildRecblock(int width, int height, string order)
    {
        IntPtr ptr = Marshal.AllocHGlobal(12);
        for (int i = 0; i < 12; i++)
        {
            Marshal.WriteByte(ptr, i, 0);
        }
        WriteRecblock(ptr, 0, 0, width - 1, height - 1, order);
        Marshal.WriteIntPtr(ptr, 8, IntPtr.Zero);
        return ptr;
    }

    private static IntPtr BuildRecblockChain(string path, string order)
    {
        string[] lines = File.ReadAllLines(path);
        IntPtr head = IntPtr.Zero;
        IntPtr prev = IntPtr.Zero;
        foreach (string line in lines)
        {
            if (line.Trim().Length == 0 || line.StartsWith("#"))
            {
                continue;
            }
            string[] parts = line.Split('\t');
            IntPtr ptr = Marshal.AllocHGlobal(12);
            for (int i = 0; i < 12; i++)
            {
                Marshal.WriteByte(ptr, i, 0);
            }
            WriteRecblock(ptr, int.Parse(parts[0]), int.Parse(parts[1]), int.Parse(parts[2]), int.Parse(parts[3]), order);
            Marshal.WriteIntPtr(ptr, 8, IntPtr.Zero);
            if (head == IntPtr.Zero)
            {
                head = ptr;
            }
            if (prev != IntPtr.Zero)
            {
                Marshal.WriteIntPtr(prev, 8, ptr);
            }
            prev = ptr;
        }
        return head;
    }

    private static void WriteRecblock(IntPtr ptr, int left, int top, int right, int bottom, string order)
    {
        if (order == "tbrl")
        {
            Marshal.WriteInt16(ptr, 0, (short)top);
            Marshal.WriteInt16(ptr, 2, (short)bottom);
            Marshal.WriteInt16(ptr, 4, (short)left);
            Marshal.WriteInt16(ptr, 6, (short)right);
            return;
        }
        Marshal.WriteInt16(ptr, 0, (short)left);
        Marshal.WriteInt16(ptr, 2, (short)top);
        Marshal.WriteInt16(ptr, 4, (short)right);
        Marshal.WriteInt16(ptr, 6, (short)bottom);
    }

    private static void FreeRecblocks(IntPtr head)
    {
        IntPtr current = head;
        while (current != IntPtr.Zero)
        {
            IntPtr next = Marshal.ReadIntPtr(current, 8);
            Marshal.FreeHGlobal(current);
            current = next;
        }
    }

    private static byte[] PackImage(string path, string imageFormat, out int width, out int height)
    {
        using (Bitmap bitmap = new Bitmap(path))
        {
            width = bitmap.Width;
            height = bitmap.Height;
            if (imageFormat == "gray")
            {
                byte[] gray = new byte[width * height];
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        Color pixel = bitmap.GetPixel(x, y);
                        gray[y * width + x] = (byte)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                    }
                }
                return gray;
            }

            int packedWidth = imageFormat == "packed16" ? ((width + 15) & -16) : width;
            int stride = (packedWidth + 7) / 8;
            byte[] packed = new byte[stride * height];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Color pixel = bitmap.GetPixel(x, y);
                    int gray = (int)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                    if (gray < 180)
                    {
                        packed[y * stride + x / 8] |= Masks[x % 8];
                    }
                }
            }
            return packed;
        }
    }

    private static string BuildJson(
        string imagePath,
        int init,
        int width,
        int height,
        string call,
        string imageFormat,
        IntPtr regionInfo,
        string error,
        int recognizeReturn,
        ushort[] recognizeTextA,
        ushort[] recognizeTextB,
        short[] recognizeMetrics,
        bool engStrRegion)
    {
        StringBuilder sb = new StringBuilder();
        sb.Append("{\n");
        sb.AppendFormat("  \"image_path\": \"{0}\",\n", Escape(imagePath));
        sb.AppendFormat("  \"init\": {0},\n", init);
        sb.AppendFormat("  \"width\": {0},\n", width);
        sb.AppendFormat("  \"height\": {0},\n", height);
        sb.AppendFormat("  \"call\": \"{0}\",\n", Escape(call));
        sb.AppendFormat("  \"image_format\": \"{0}\",\n", Escape(imageFormat));
        sb.AppendFormat("  \"region_ptr\": \"0x{0:x8}\",\n", regionInfo.ToInt32());
        sb.AppendFormat("  \"error\": \"{0}\",\n", Escape(error));
        sb.AppendFormat("  \"recognize_return\": {0},\n", recognizeReturn);
        AppendUshortArray(sb, "recognize_text_a_codes", recognizeTextA, 80);
        sb.Append(",\n");
        AppendUshortArray(sb, "recognize_text_b_codes", recognizeTextB, 80);
        sb.Append(",\n");
        AppendShortArray(sb, "recognize_metrics", recognizeMetrics, 80);
        sb.Append(",\n");
        sb.Append("  \"lines\": [\n");
        if (engStrRegion)
        {
            AppendEngStrLines(sb, regionInfo);
        }
        else
        {
            int lineIndex = 0;
            IntPtr line = regionInfo;
            while (line != IntPtr.Zero && lineIndex < 256)
            {
                if (lineIndex > 0)
                {
                    sb.Append(",\n");
                }
                AppendLine(sb, line, lineIndex);
                line = ReadPtr(line, 28);
                lineIndex++;
            }
        }
        sb.Append("\n  ]\n}\n");
        return sb.ToString();
    }

    private static void AppendEngStrLines(StringBuilder sb, IntPtr regionInfo)
    {
        int lineIndex = 0;
        IntPtr region = regionInfo;
        while (region != IntPtr.Zero && lineIndex < 256)
        {
            IntPtr line = ReadPtr(region, 4);
            while (line != IntPtr.Zero && lineIndex < 256)
            {
                if (lineIndex > 0)
                {
                    sb.Append(",\n");
                }
                AppendEngStrLine(sb, line, lineIndex);
                line = ReadPtr(line, 0x10);
                lineIndex++;
            }
            region = ReadPtr(region, 0x10);
        }
    }

    private static void AppendEngStrLine(StringBuilder sb, IntPtr line, int lineIndex)
    {
        sb.Append("    {\n");
        sb.AppendFormat("      \"index\": {0},\n", lineIndex);
        sb.AppendFormat("      \"raw_count\": {0},\n", ReadInt16(line, 0));
        sb.AppendFormat("      \"flags\": {0},\n", ReadInt32(line, 0x0c));
        sb.Append("      \"groups\": [\n");
        IntPtr group = ReadPtr(line, 4);
        int groupIndex = 0;
        while (group != IntPtr.Zero && groupIndex < 512)
        {
            if (groupIndex > 0)
            {
                sb.Append(",\n");
            }
            AppendEngStrGroup(sb, group, groupIndex);
            group = ReadPtr(group, 0x18);
            groupIndex++;
        }
        sb.Append("\n      ]\n    }");
    }

    private static void AppendEngStrGroup(StringBuilder sb, IntPtr group, int groupIndex)
    {
        sb.Append("        {\n");
        sb.AppendFormat("          \"index\": {0},\n", groupIndex);
        sb.AppendFormat("          \"raw_count\": {0},\n", ReadInt16(group, 0));
        sb.Append("          \"chars\": [");
        IntPtr chr = ReadPtr(group, 4);
        int charIndex = 0;
        while (chr != IntPtr.Zero && charIndex < 512)
        {
            if (charIndex > 0)
            {
                sb.Append(", ");
            }
            AppendEngStrChar(sb, chr, charIndex);
            chr = ReadPtr(chr, 0x38);
            charIndex++;
        }
        sb.Append("]\n        }");
    }

    private static void AppendEngStrChar(StringBuilder sb, IntPtr chr, int charIndex)
    {
        ushort code = ReadUInt16(chr, 2);
        ushort score = ReadUInt16(chr, 0x16);
        int top = ReadUInt16(chr, 0x2a);
        int bottom = ReadUInt16(chr, 0x2c);
        int left = ReadUInt16(chr, 0x2e);
        int right = ReadUInt16(chr, 0x30);
        sb.Append("{");
        sb.AppendFormat("\"index\":{0},", charIndex);
        AppendRect(sb, "bbox", left, top, right, bottom);
        sb.AppendFormat(",\"flags\":0,\"candidate_count\":1,\"codes\":[{0}],\"scores\":[{1}]", code, score);
        sb.Append("}");
    }

    private static void AppendUshortArray(StringBuilder sb, string name, ushort[] values, int limit)
    {
        sb.AppendFormat("  \"{0}\": [", name);
        int count = Math.Min(values.Length, limit);
        for (int i = 0; i < count; i++)
        {
            if (i > 0)
            {
                sb.Append(",");
            }
            sb.Append(values[i]);
        }
        sb.Append("]");
    }

    private static void AppendShortArray(StringBuilder sb, string name, short[] values, int limit)
    {
        sb.AppendFormat("  \"{0}\": [", name);
        int count = Math.Min(values.Length, limit);
        for (int i = 0; i < count; i++)
        {
            if (i > 0)
            {
                sb.Append(",");
            }
            sb.Append(values[i]);
        }
        sb.Append("]");
    }

    private static void AppendLine(StringBuilder sb, IntPtr line, int lineIndex)
    {
        sb.Append("    {\n");
        sb.AppendFormat("      \"index\": {0},\n", lineIndex);
        sb.AppendFormat("      \"raw_count\": {0},\n", ReadInt16(line, 0));
        sb.AppendFormat("      \"flags\": {0},\n", ReadInt32(line, 24));
        AppendRect(sb, "bbox", ReadInt32(line, 8), ReadInt32(line, 12), ReadInt32(line, 16), ReadInt32(line, 20));
        sb.Append(",\n      \"groups\": [\n");
        IntPtr group = ReadPtr(line, 4);
        int groupIndex = 0;
        while (group != IntPtr.Zero && groupIndex < 512)
        {
            if (groupIndex > 0)
            {
                sb.Append(",\n");
            }
            AppendGroup(sb, group, groupIndex);
            group = ReadPtr(group, 24);
            groupIndex++;
        }
        sb.Append("\n      ]\n    }");
    }

    private static void AppendGroup(StringBuilder sb, IntPtr group, int groupIndex)
    {
        sb.Append("        {\n");
        sb.AppendFormat("          \"index\": {0},\n", groupIndex);
        sb.AppendFormat("          \"raw_count\": {0},\n", ReadInt16(group, 0));
        AppendRect(sb, "bbox", ReadInt32(group, 8), ReadInt32(group, 12), ReadInt32(group, 16), ReadInt32(group, 20));
        sb.Append(",\n          \"chars\": [");
        IntPtr chr = ReadPtr(group, 4);
        int charIndex = 0;
        while (chr != IntPtr.Zero && charIndex < 512)
        {
            if (charIndex > 0)
            {
                sb.Append(", ");
            }
            AppendChar(sb, chr, charIndex);
            chr = ReadPtr(chr, 64);
            charIndex++;
        }
        sb.Append("]\n        }");
    }

    private static void AppendChar(StringBuilder sb, IntPtr chr, int charIndex)
    {
        int count = Math.Max(0, Math.Min(10, (int)ReadInt16(chr, 0)));
        sb.Append("{");
        sb.AppendFormat("\"index\":{0},", charIndex);
        AppendRect(sb, "bbox", ReadInt32(chr, 44), ReadInt32(chr, 48), ReadInt32(chr, 52), ReadInt32(chr, 56));
        sb.AppendFormat(",\"flags\":{0},\"candidate_count\":{1},\"codes\":[", ReadInt32(chr, 60), count);
        for (int i = 0; i < count; i++)
        {
            if (i > 0)
            {
                sb.Append(",");
            }
            sb.Append(ReadUInt16(chr, 4 + i * 2));
        }
        sb.Append("],\"scores\":[");
        for (int i = 0; i < count; i++)
        {
            if (i > 0)
            {
                sb.Append(",");
            }
            sb.Append(ReadUInt16(chr, 24 + i * 2));
        }
        sb.Append("]}");
    }

    private static void AppendRect(StringBuilder sb, string name, int left, int top, int right, int bottom)
    {
        sb.AppendFormat("\"{0}\": {{\"left\": {1}, \"top\": {2}, \"right\": {3}, \"bottom\": {4}, \"width\": {5}, \"height\": {6}}}", name, left, top, right, bottom, right - left + 1, bottom - top + 1);
    }

    private static IntPtr ReadPtr(IntPtr ptr, int offset)
    {
        return Marshal.ReadIntPtr(ptr, offset);
    }

    private static short ReadInt16(IntPtr ptr, int offset)
    {
        return Marshal.ReadInt16(ptr, offset);
    }

    private static ushort ReadUInt16(IntPtr ptr, int offset)
    {
        return (ushort)Marshal.ReadInt16(ptr, offset);
    }

    private static int ReadInt32(IntPtr ptr, int offset)
    {
        return Marshal.ReadInt32(ptr, offset);
    }

    private static string Escape(string value)
    {
        return value.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }

    private static int OnProgress(uint current)
    {
        return 0;
    }

    private static int OnCancel()
    {
        return 0;
    }
}
