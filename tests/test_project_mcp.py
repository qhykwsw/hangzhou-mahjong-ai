"""Repository access boundaries and a real MCP stdio round trip."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mahjong_ai.project_mcp.reader import MAX_FILE, RepositoryReader, allowed, bounded_result, redact

ROOT = Path(__file__).resolve().parents[1]


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL).decode().strip()


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    git(root, 'init', '-q')
    (root / 'README.md').write_text('# Demo\ncache logic\npassword = fixture-secret\n')
    (root / 'src/demo').mkdir(parents=True)
    (root / 'src/demo/main.py').write_text('cache = True\n')
    (root / 'src/demo/fast.c').write_text('/* cache */\n')
    git(root, 'add', '.')
    git(root, '-c', 'user.name=Test', '-c', 'user.email=test@example.test', 'commit', '-qm', 'initial cache')
    return RepositoryReader(root)


@pytest.mark.parametrize('path', [
    '../README.md', '/etc/passwd', 'src/../README.md', 'src//main.py', 'src/./main.py',
    '.env', 'src/.hidden/test.py', '.local/research/data.json', 'src/credentials/test.json',
    'src/private.key', 'src/password.json', 'projects/demo/models/model.json',
    'projects/demo/releases/version.json', 'projects/demo/runs/report.json',
    'docs/archive/old.md', 'src/a_bak/test.py', 'src/test.sqlite', 'build/result.txt',
])
def test_denied_paths(path):
    assert not allowed(path)


def test_current_historical_and_untracked_reads(repo):
    head = repo.revision('HEAD')
    (repo.root / 'README.md').write_text('# Changed\n')
    assert 'Changed' in repo.read('README.md', 1, 10, None)['text']
    result = repo.read('README.md', 1, 2, head)
    assert result['revision'] == head and result['next_line'] == 3
    assert 'Demo' in result['text']
    assert head in repo.history('README.md', 1)['history']
    (repo.root / 'src/demo/untracked.py').write_text('hidden')
    with pytest.raises(ValueError):
        repo.read('src/demo/untracked.py', 1, 10, None)
    assert allowed('src/demo/fast.c')
    with pytest.raises(ValueError):
        repo.read('README.md', 1, 10, '--output=/tmp/unsafe')


@pytest.mark.parametrize('kind', ['symlink', 'parent_symlink', 'binary', 'large', 'fifo'])
def test_content_guards(repo, tmp_path, kind):
    path = repo.root / 'src/demo/main.py'
    path.unlink()
    if kind == 'symlink':
        path.symlink_to(repo.root / 'README.md')
    elif kind == 'parent_symlink':
        (repo.root / 'src/demo/fast.c').unlink()
        path.parent.rmdir()
        (tmp_path / 'main.py').write_text('outside')
        path.parent.symlink_to(tmp_path, target_is_directory=True)
    elif kind == 'binary':
        path.write_bytes(b'bad\x00binary')
    elif kind == 'large':
        path.write_bytes(b'x' * (MAX_FILE + 1))
    else:
        os.mkfifo(path)
    with pytest.raises((OSError, ValueError)):
        repo.read('src/demo/main.py', 1, 10, None)
    assert not repo.search('outside', 'src/', 0)['results']


def test_historical_symlink_and_deleted_file(repo):
    path = repo.root / 'src/demo/main.py'
    path.unlink()
    path.symlink_to('fast.c')
    git(repo.root, 'add', '.')
    git(repo.root, '-c', 'user.name=Test', '-c', 'user.email=test@example.test', 'commit', '-qm', 'link')
    with pytest.raises(ValueError):
        repo.read('src/demo/main.py', 1, 10, 'HEAD')
    first = git(repo.root, 'rev-parse', 'HEAD~1')
    assert 'cache' in repo.read('src/demo/main.py', 1, 10, first)['text']


def test_redaction_and_search_pagination(repo):
    assert not repo.search('fixture-secret', '', 0)['results']
    pem = 'before\n-----BEGIN PRIVATE KEY-----\nFAKE_PRIVATE\n-----END PRIVATE KEY-----\nafter'
    assert 'FAKE_PRIVATE' not in redact(pem)
    assert redact(pem).count('\n') == pem.count('\n')
    with pytest.raises(ValueError):
        bounded_result({'text': 'x' * 100000})
    for i in range(105):
        (repo.root / f'src/demo/file{i:03}.py').write_text('match here\n')
    git(repo.root, 'add', '.')
    offset, found = 0, []
    while offset is not None:
        page = repo.search('match', 'src/demo/file', offset)
        found.extend(item['path'] for item in page['results'])
        offset = page['next_offset']
    assert len(found) == len(set(found)) == 105


def test_stdio_protocol_and_audit(repo, tmp_path):
    async def check():
        audit = tmp_path / 'audit.jsonl'
        params = StdioServerParameters(command=sys.executable, args=[
            '-m', 'mahjong_ai.project_mcp', '--workspace-root', str(repo.root), '--audit-path', str(audit),
        ], cwd=str(ROOT))
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = (await client.list_tools()).tools
                assert {t.name for t in tools} == {'project_overview', 'list_files', 'read_file', 'search', 'git_history'}
                assert all(t.annotations.readOnlyHint and not t.annotations.destructiveHint and t.outputSchema for t in tools)
                overview = await client.call_tool('project_overview', {})
                assert not overview.isError
                head = overview.structuredContent['head']
                listing = await client.call_tool('list_files', {'limit': 1, 'revision': head})
                assert listing.structuredContent['next_offset'] == 1
                readout = await client.call_tool('read_file', {'path': 'README.md'})
                assert 'fixture-secret' not in str(readout)
                assert 'Demo' in readout.structuredContent['text']
                search = await client.call_tool('search', {'query': 'cache', 'prefix': 'src/'})
                assert search.structuredContent['results']
                history = await client.call_tool('git_history', {'path': 'README.md'})
                assert head in history.structuredContent['history']
                for name, args in [
                    ('read_file', {'path': '../private-secret.txt'}),
                    ('read_file', {'path': 'README.md', 'lines': 99999}),
                    ('read_file', {'path': 'README.md', 'revision': '-unsafe'}),
                    ('runtime_query', {}), ('research_create_candidate', {}),
                ]:
                    assert (await client.call_tool(name, args)).isError
        logs = [json.loads(line) for line in audit.read_text().splitlines()]
        assert all(set(row) == {'at', 'tool', 'status', 'elapsed_ms'} for row in logs)
        assert 'private-secret' not in audit.read_text() and 'fixture-secret' not in audit.read_text()
        assert audit.stat().st_mode & 0o777 == 0o600
    asyncio.run(check())


def test_audit_failure_does_not_echo_local_paths(tmp_path):
    from mcp.server.fastmcp.exceptions import ToolError
    from mahjong_ai.project_mcp.server import Audit
    import time
    forbidden = tmp_path / 'private-secret-location'
    forbidden.write_text('not a directory')
    with pytest.raises(ToolError) as error:
        Audit(forbidden / 'audit.jsonl').record('read_file', 'started', time.monotonic())
    assert 'private-secret-location' not in str(error.value)
