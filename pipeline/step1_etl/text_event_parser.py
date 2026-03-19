import ast
import json
import re
from pathlib import Path
from typing import Iterable, Optional, TextIO

SELECTED_ATTRIBUTES = {
    "ProcessID",
    "ProcessId",
    "CreateTime",
    "ThreadId",
    "ThreadID",
    "FrozenProcessID",
    "Task Name",
    "TimeStamp",
    "UserStackLimit",
    "StackLimit",
    "Win32StartAddr",
    "TebBase",
    "OldPriority",
    "NewPriority",
    "StackBase",
    "SubProcessStack",
    "UserStackBase",
    "StartAddr",
    "Job",
    "ID",
    "ContainerID",
    "FileObject",
    "FileKey",
    "FilePath",
    "ByteOffset",
    "IOFlags",
    "IOSize",
    "Irp",
    "daddr",
    "saddr",
    "sport",
    "dport",
    "connid",
    "ParentProcessID",
    "ImageSize",
    "ImageCheckSum",
    "HandleCount",
    "ImageBase",
    "ImageName",
    "TimeDateStamp",
    "ExitCode",
    "SessionId",
    "ExitTime",
    "ValueName",
    "KeyObject",
    "RelativeName",
    "Index",
    "FileName",
}

EXCLUDED_ATTRIBUTES = {
    "Keyword",
    "Flags",
    "Description",
}

TASK_FIELDS = {"Task Name"}

TASK_NAMES = {
    "CLEANUP",
    "CLOSE",
    "CREATE",
    "CREATENEWFILE",
    "DELETEPATH",
    "DIRENUM",
    "DIRNOTIFY",
    "FLUSH",
    "FSCTL",
    "NAMECREATE",
    "NAMEDELETE",
    "OPERATIONEND",
    "QUERYINFO",
    "QUERYINFORMATION",
    "QUERYEA",
    "QUERYSECURITY",
    "READ",
    "WRITE",
    "SETDELETE",
    "SETINFORMATION",
    "PAGEPRIORITYCHANGE",
    "IOPRIORITYCHANGE",
    "CPUBASEPRIORITYCHANGE",
    "IMAGEPRIORITYCHANGE",
    "CPUPRIORITYCHANGE",
    "IMAGELOAD",
    "IMAGEUNLOAD",
    "PROCESSSTOP",
    "PROCESSSTART",
    "PROCESSFREEZE",
    "PSDISKIOATTRIBUTE",
    "PSIORATECONTROL",
    "KERNEL_NETWORK_TASK_UDPIP",
    "KERNEL_NETWORK_TASK_TCPIP",
    "MICROSOFT-WINDOWS-KERNEL-REGISTRY",
    "THREADSTART",
    "THREADSTOP",
    "THREADWORKONBEHALFUPDATE",
    "JOBSTART",
    "JOBTERMINATE",
    "LOSTEVENT",
    "PSDISKIOATTRIBUTION",
    "RENAME",
    "RENAMEPATH",
    "THIS GROUP OF EVENTS TRACKS THE PERFORMANCE OF FLUSHING HIVES",
}

HEX_ATTRIBUTES = {
    "UserStackLimit",
    "StackLimit",
    "Win32StartAddr",
    "TebBase",
    "StackBase",
    "SubProcessStack",
    "UserStackBase",
    "StartAddr",
    "ByteOffset",
    "IOSize",
    "Irp",
    "ImageSize",
    "ImageBase",
    "FileObject",
    "FileKey",
    "IOFlags",
}

IP_ADDRESS_FIELDS = {"daddr", "saddr"}
TIME_FIELDS = {"ExitTime", "CreateTime"}

PROVIDER_ID_MAP = {
    "{70EB4F03-C1DE-4F73-A051-33D13D5413BD}": 1,
    "{22FB2CD6-0E7B-422B-A0C7-2FAD1FD0E716}": 2,
    "{7DD42A49-5329-4832-8DFD-43D979153A88}": 3,
    "{EDD08927-9CC4-4E65-B970-C2560FB5C289}": 4,
}


def read_marshaled_line(file_obj: TextIO) -> str:
    """
    Read a single legacy ETW text line and strip the trailing marshal suffix that
    appears in the original data dump format.
    """
    line = file_obj.readline()
    if not line:
        return ""

    json_str = json.dumps(line)
    cleaned = re.sub(r"\)\(([0-9]*)", "", json_str)
    return json.loads(cleaned)


def parse_legacy_tuple_line(line: str):
    """
    Parse a legacy line of the form:
        (<ignored_prefix>, <event_dict>)
    and return the event dictionary.
    """
    parsed = ast.literal_eval(line)
    if not isinstance(parsed, tuple) or len(parsed) < 2:
        raise ValueError("Expected a tuple-like legacy ETW line with an event record in position 1.")
    return parsed[1]


def convert_text_log_to_event_lines(input_path: str | Path, output_path: str | Path) -> None:
    """
    Convert legacy raw ETW text logs into one event-record-per-line output.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with input_path.open("r", encoding="utf-8", errors="ignore") as src, output_path.open(
        "w", encoding="utf-8", errors="ignore"
    ) as dst:
        for raw_line in src:
            if raw_line == "\n":
                continue

            line = raw_line
            if "||" in line:
                line = line[: line.find("||")]

            record = parse_legacy_tuple_line(line)
            dst.write(f"{record}\n")


def addr2dec(addr: str) -> int | None:
    temp = addr.split(".")
    x = int(temp[0])
    x2 = int(temp[1])

    if 0 <= x <= 127:
        if x == 10:
            return 6
        if x == 127:
            return 7
        return 1
    if 128 <= x < 191:
        if x == 172 and 16 <= x2 <= 31:
            return 8
        return 2
    if 192 <= x <= 223:
        if x == 192 and x2 == 168:
            return 9
        return 3
    if 224 <= x <= 239:
        return 4
    if 240 <= x <= 254:
        return 5
    if x == 255:
        return 10
    return None


def check_ipv6(ipv6: str) -> bool:
    return bool(
        re.match(
            r"^\s*(((([0-9A-Fa-f]{1,4}:){7}([0-9A-Fa-f]{1,4}|:))|"
            r"(([0-9A-Fa-f]{1,4}:){6}(:[0-9A-Fa-f]{1,4}|"
            r"((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
            r"(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|"
            r"(([0-9A-Fa-f]{1,4}:){5}(((:[0-9A-Fa-f]{1,4}){1,2})|"
            r":((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
            r"(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3})|:))|"
            r"(([0-9A-Fa-f]{1,4}:){4}(((:[0-9A-Fa-f]{1,4}){1,3})|"
            r"((:[0-9A-Fa-f]{1,4})?:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
            r"(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|"
            r"(([0-9A-Fa-f]{1,4}:){3}(((:[0-9A-Fa-f]{1,4}){1,4})|"
            r"((:[0-9A-Fa-f]{1,4}){0,2}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
            r"(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|"
            r"(([0-9A-Fa-f]{1,4}:){2}(((:[0-9A-Fa-f]{1,4}){1,5})|"
            r"((:[0-9A-Fa-f]{1,4}){0,3}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
            r"(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|"
            r"(([0-9A-Fa-f]{1,4}:){1}(((:[0-9A-Fa-f]{1,4}){1,6})|"
            r"((:[0-9A-Fa-f]{1,4}){0,4}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
            r"(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:))|"
            r"(:(((:[0-9A-Fa-f]{1,4}){1,7})|"
            r"((:[0-9A-Fa-f]{1,4}){0,5}:((25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
            r"(\.(25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}))|:)))(%.+)?\s*$",
            ipv6,
        )
    )


def ipseg2str(ipseglist: Iterable[str]) -> str:
    ipstr = ""
    for ipseg in ipseglist:
        if len(ipseg) == 1:
            ipseg = "000" + ipseg
        elif len(ipseg) == 2:
            ipseg = "00" + ipseg
        elif len(ipseg) == 3:
            ipseg = "0" + ipseg
        elif len(ipseg) != 4:
            return ""
        ipstr += ipseg
    return ipstr


def no_compress_ipv6_to_dec(ipv6: str) -> int | str:
    iplist = ipv6.split(":")
    if iplist:
        return int(ipseg2str(iplist), 16)
    return ""


def compress_ipv6_to_dec(ipv6: str) -> int | str:
    compress_list = ipv6.split("::")
    ipstr = ""
    part1 = []
    part2 = []
    if len(compress_list) == 2:
        part1 = compress_list[0].split(":") if compress_list[0] else []
        part2 = compress_list[1].split(":") if compress_list[1] else []

    if part1 or part2:
        if part1:
            ipstr += ipseg2str(part1)
            for _ in range(8 - len(part1) - len(part2)):
                ipstr += "0000"
            ipstr += ipseg2str(part2)
            return int(ipstr, 16)
        ipstr = "".join(part2)
        return int(ipstr)

    return ""


def ipv6_to_dec(ipv6: str) -> int | str | None:
    if not check_ipv6(ipv6):
        return None

    has_compression = "::" in ipv6
    has_ipv4_tail = "." in ipv6

    if not has_compression and not has_ipv4_tail:
        return no_compress_ipv6_to_dec(ipv6)
    if has_compression and not has_ipv4_tail:
        return compress_ipv6_to_dec(ipv6)
    return None


def main() -> None:
    raise SystemExit("Use convert_text_log_to_event_lines(input_path, output_path).")


if __name__ == "__main__":
    main()
