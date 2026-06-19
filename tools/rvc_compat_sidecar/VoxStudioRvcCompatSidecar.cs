using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;

internal static class VoxStudioRvcCompatSidecar
{
    private const string ServerName = "VoxStudioRvcCompatSidecar";

    private static int Main(string[] args)
    {
        string host = "127.0.0.1";
        int port = 18888;

        for (int index = 0; index < args.Length; index++)
        {
            if (args[index] == "--host" && index + 1 < args.Length)
            {
                host = args[++index];
            }
            else if (args[index] == "--port" && index + 1 < args.Length)
            {
                int parsedPort;
                if (int.TryParse(args[++index], out parsedPort))
                {
                    port = parsedPort;
                }
            }
        }

        IPAddress address;
        if (!IPAddress.TryParse(host, out address))
        {
            Console.Error.WriteLine("Invalid host: " + host);
            return 2;
        }

        var listener = new TcpListener(address, port);
        listener.Start();
        Console.WriteLine(ServerName + " listening on http://" + host + ":" + port);

        while (true)
        {
            TcpClient client = listener.AcceptTcpClient();
            ThreadPool.QueueUserWorkItem(_ => HandleClient(client));
        }
    }

    private static void HandleClient(TcpClient client)
    {
        using (client)
        using (NetworkStream stream = client.GetStream())
        {
            try
            {
                string requestLine = ReadLine(stream);
                if (string.IsNullOrEmpty(requestLine))
                {
                    return;
                }

                string[] requestParts = requestLine.Split(' ');
                if (requestParts.Length < 2)
                {
                    WriteJson(stream, 400, "{\"ok\":false,\"message\":\"Bad request.\"}");
                    return;
                }

                var headers = ReadHeaders(stream);
                string method = requestParts[0].ToUpperInvariant();
                string path = requestParts[1];

                if (method == "GET" && path == "/health")
                {
                    WriteJson(
                        stream,
                        200,
                        "{\"ok\":true,\"engine\":\"compat-pass-through\",\"cuda_available\":false,\"cuda_version\":\"n/a\",\"loaded_model_id\":\"pass-through\",\"last_latency_ms\":0,\"message\":\"Compatibility sidecar running. Audio is passed through for testing; no real RVC conversion is performed.\"}");
                    return;
                }

                if (method == "POST" && path == "/convert_chunk")
                {
                    byte[] body = ReadBody(stream, headers);
                    byte[] audio = ExtractMultipartAudio(headers, body);
                    if (audio.Length == 0)
                    {
                        WriteJson(stream, 400, "{\"ok\":false,\"message\":\"No audio part found.\"}");
                        return;
                    }

                    WriteBytes(stream, 200, "audio/L16", audio);
                    return;
                }

                WriteJson(stream, 404, "{\"ok\":false,\"message\":\"Not found.\"}");
            }
            catch (Exception exception)
            {
                try
                {
                    WriteJson(stream, 500, "{\"ok\":false,\"message\":\"" + JsonEscape(exception.Message) + "\"}");
                }
                catch
                {
                    // The client disconnected before the error could be reported.
                }
            }
        }
    }

    private static Dictionary<string, string> ReadHeaders(Stream stream)
    {
        var headers = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        while (true)
        {
            string line = ReadLine(stream);
            if (line.Length == 0)
            {
                return headers;
            }

            int colon = line.IndexOf(':');
            if (colon <= 0)
            {
                continue;
            }

            string name = line.Substring(0, colon).Trim();
            string value = line.Substring(colon + 1).Trim();
            headers[name] = value;
        }
    }

    private static string ReadLine(Stream stream)
    {
        var bytes = new List<byte>();
        while (true)
        {
            int value = stream.ReadByte();
            if (value < 0)
            {
                break;
            }

            if (value == '\n')
            {
                break;
            }

            if (value != '\r')
            {
                bytes.Add((byte)value);
            }
        }

        return Encoding.ASCII.GetString(bytes.ToArray());
    }

    private static byte[] ReadBody(Stream stream, Dictionary<string, string> headers)
    {
        string contentLengthText;
        if (!headers.TryGetValue("Content-Length", out contentLengthText))
        {
            return new byte[0];
        }

        int contentLength;
        if (!int.TryParse(contentLengthText, out contentLength) || contentLength <= 0)
        {
            return new byte[0];
        }

        byte[] body = new byte[contentLength];
        int offset = 0;
        while (offset < body.Length)
        {
            int read = stream.Read(body, offset, body.Length - offset);
            if (read <= 0)
            {
                break;
            }

            offset += read;
        }

        if (offset == body.Length)
        {
            return body;
        }

        byte[] partial = new byte[offset];
        Buffer.BlockCopy(body, 0, partial, 0, offset);
        return partial;
    }

    private static byte[] ExtractMultipartAudio(Dictionary<string, string> headers, byte[] body)
    {
        string contentType;
        if (!headers.TryGetValue("Content-Type", out contentType))
        {
            return body;
        }

        string boundary = BoundaryFromContentType(contentType);
        if (string.IsNullOrEmpty(boundary))
        {
            return body;
        }

        string marker = "--" + boundary;
        byte[] markerBytes = Encoding.ASCII.GetBytes(marker);
        byte[] headerSeparator = Encoding.ASCII.GetBytes("\r\n\r\n");
        byte[] nextMarker = Encoding.ASCII.GetBytes("\r\n" + marker);

        int searchStart = 0;
        while (true)
        {
            int markerIndex = IndexOf(body, markerBytes, searchStart);
            if (markerIndex < 0)
            {
                return new byte[0];
            }

            int headersStart = markerIndex + markerBytes.Length;
            if (headersStart + 2 <= body.Length && body[headersStart] == '-' && body[headersStart + 1] == '-')
            {
                return new byte[0];
            }

            if (headersStart + 2 <= body.Length && body[headersStart] == '\r' && body[headersStart + 1] == '\n')
            {
                headersStart += 2;
            }

            int dataStart = IndexOf(body, headerSeparator, headersStart);
            if (dataStart < 0)
            {
                return new byte[0];
            }

            string partHeaders = Encoding.ASCII.GetString(body, headersStart, dataStart - headersStart);
            dataStart += headerSeparator.Length;

            int dataEnd = IndexOf(body, nextMarker, dataStart);
            if (dataEnd < 0)
            {
                return new byte[0];
            }

            if (partHeaders.IndexOf("name=\"audio\"", StringComparison.OrdinalIgnoreCase) >= 0)
            {
                byte[] audio = new byte[dataEnd - dataStart];
                Buffer.BlockCopy(body, dataStart, audio, 0, audio.Length);
                return audio;
            }

            searchStart = dataEnd + 2;
        }
    }

    private static string BoundaryFromContentType(string contentType)
    {
        string[] pieces = contentType.Split(';');
        foreach (string piece in pieces)
        {
            string trimmed = piece.Trim();
            if (trimmed.StartsWith("boundary=", StringComparison.OrdinalIgnoreCase))
            {
                string value = trimmed.Substring("boundary=".Length).Trim();
                return value.Trim('"');
            }
        }

        return string.Empty;
    }

    private static int IndexOf(byte[] haystack, byte[] needle, int start)
    {
        if (needle.Length == 0 || haystack.Length < needle.Length)
        {
            return -1;
        }

        for (int index = Math.Max(0, start); index <= haystack.Length - needle.Length; index++)
        {
            bool match = true;
            for (int needleIndex = 0; needleIndex < needle.Length; needleIndex++)
            {
                if (haystack[index + needleIndex] != needle[needleIndex])
                {
                    match = false;
                    break;
                }
            }

            if (match)
            {
                return index;
            }
        }

        return -1;
    }

    private static void WriteJson(Stream stream, int statusCode, string json)
    {
        WriteBytes(stream, statusCode, "application/json", Encoding.UTF8.GetBytes(json));
    }

    private static void WriteBytes(Stream stream, int statusCode, string contentType, byte[] body)
    {
        string reason = statusCode == 200 ? "OK" : "Error";
        string header =
            "HTTP/1.1 " + statusCode + " " + reason + "\r\n" +
            "Server: " + ServerName + "\r\n" +
            "Content-Type: " + contentType + "\r\n" +
            "Content-Length: " + body.Length + "\r\n" +
            "Connection: close\r\n\r\n";
        byte[] headerBytes = Encoding.ASCII.GetBytes(header);
        stream.Write(headerBytes, 0, headerBytes.Length);
        stream.Write(body, 0, body.Length);
    }

    private static string JsonEscape(string text)
    {
        return text.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }
}
