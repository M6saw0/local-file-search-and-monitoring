#!/usr/bin/env python3

"""
Hybrid Search MCP Server - FastMCP クライアントテスト

FastMCPのクライアント機能を使用したMCPサーバーのテスト
公式ドキュメント: https://gofastmcp.com/clients/client
"""

import asyncio
import argparse
import logging
from typing import Dict, Any

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def test_fastmcp_server(host: str = "127.0.0.1", port: str = "8000", query: str = "Python ファイル処理", mode: str = "hybrid", debug: bool = False, test_file_path: str = None):
    """
    FastMCPサーバーをテストします
    
    Args:
        query: テストクエリ
        mode: 検索モード
        debug: デバッグモード
    """
    try:
        # FastMCPクライアントをインポート
        from fastmcp import Client
        
        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        
        print("🚀 FastMCP クライアントテスト開始")
        print(f"📝 テストクエリ: '{query}'")
        print(f"🔧 検索モード: {mode}")
        print("-" * 60)
        
        # HTTPサーバーに接続 (FastMCPのデフォルト: http://127.0.0.1:8000)
        # FastMCPのデフォルトポートは8000のようなので、まずそれを試す
        server_url = f"http://{host}:{port}/mcp"
        logger.info(f"🌐 FastMCPサーバーに接続中: {server_url}")
        
        client = Client(server_url)
        
        async with client:
            # 1. サーバー接続確認
            print("1️⃣ サーバー接続確認...")
            try:
                await client.ping()
                print("✅ サーバー接続成功")
            except Exception as e:
                print(f"❌ サーバー接続失敗: {e}")
                # デフォルトポートで失敗した場合、8080も試してみる
                server_url = "http://127.0.0.1:8080"
                logger.info(f"🔄 代替ポートで再試行: {server_url}")
                client = Client(server_url)
                async with client:
                    await client.ping()
                    print("✅ 代替ポートでサーバー接続成功")
            
            print()
            
            # 2. ツールリスト取得
            print("2️⃣ 利用可能なツール取得中...")
            try:
                tools = await client.list_tools()
                print(f"✅ ツール数: {len(tools)}個")
                
                for i, tool in enumerate(tools, 1):
                    print(f"   {i}. {tool.name}: {tool.description}")
                
                # hybrid_searchツールがあるか確認
                hybrid_search_tool = next((t for t in tools if t.name == "hybrid_search"), None)
                if not hybrid_search_tool:
                    print("❌ hybrid_searchツールが見つかりません")
                    return
                    
            except Exception as e:
                print(f"❌ ツールリスト取得失敗: {e}")
                return
            
            print()
            
            # 3. 検索テスト実行
            print("3️⃣ ハイブリッド検索テスト実行中...")
            try:
                # hybrid_searchツールを呼び出し
                result = await client.call_tool("hybrid_search", {
                    "query": query,
                    "mode": mode,
                    "max_results": 5,
                    "bm25_weight": 1.0,
                    "vector_weight": 1.0
                })
                
                print("✅ 検索テスト成功")
                print("📋 検索結果:")
                print("-" * 40)
                
                # 結果を表示
                if hasattr(result, 'content') and result.content:
                    for content in result.content:
                        if hasattr(content, 'text'):
                            print(content.text[:500])  # 最初の500文字を表示
                            if len(content.text) > 500:
                                print("... (省略)")
                else:
                    print("結果の形式が予期しないものです")
                    print(f"実際の結果: {result}")
                print("-"*50)
                print("results一覧:")
                for result in result.content:
                    print(result)
                    
            except Exception as e:
                print(f"❌ 検索テスト失敗: {e}")
                logger.debug(f"詳細エラー: {e}", exc_info=True)
                return
            
            print()
            
            # 4. 複数モードテスト
            print("4️⃣ 複数検索モードテスト実行中...")
            test_modes = ["hybrid", "bm25", "vector"]
            
            for test_mode in test_modes:
                try:
                    print(f"   🔧 {test_mode.upper()}モードテスト...")
                    
                    result = await client.call_tool("hybrid_search", {
                        "query": query,
                        "mode": test_mode,
                        "max_results": 3
                    })
                    
                    # 結果から情報を抽出（簡単な表示）
                    if hasattr(result, 'content') and result.content:
                        content_text = result.content[0].text if result.content else "結果なし"
                        # JSONパース試行
                        try:
                            import json
                            parsed = json.loads(content_text)
                            total_results = parsed.get('total_results', 0)
                            response_time = parsed.get('response_time', 0)
                            print(f"   ✅ {test_mode.upper()}: {total_results}件 ({response_time:.3f}秒)")
                        except:
                            print(f"   ✅ {test_mode.upper()}: 実行完了")
                    else:
                        print(f"   ✅ {test_mode.upper()}: 実行完了")
                        
                except Exception as e:
                    print(f"   ❌ {test_mode.upper()}: エラー - {e}")
            
            print()
            
            # 5. get_file_content ツールテスト（引数で指定されたファイル）
            if test_file_path:
                print("5️⃣ get_file_content ツールテスト実行中...")
                try:
                    print(f"   📄 対象ファイル: {test_file_path}")
                    
                    result = await client.call_tool("get_file_content", {
                        "file_path": test_file_path
                    })
                    
                    print("✅ ファイル内容取得テスト成功")
                    print("📄 取得結果:")
                    print("-" * 40)
                    
                    # 結果を表示
                    if hasattr(result, 'content') and result.content:
                        content_text = result.content[0].text if result.content else "結果なし"
                        # JSONパース試行
                        try:
                            import json
                            parsed = json.loads(content_text)
                            
                            if parsed.get('success'):
                                print(f"📁 ファイル名: {parsed.get('file_name')}")
                                print(f"📂 ファイルパス: {parsed.get('file_path')}")
                                print(f"📏 ファイルサイズ: {parsed.get('file_size')} bytes")
                                print(f"📝 テキスト長: {parsed.get('text_length')} 文字")
                                print(f"⏱️ 処理時間: {parsed.get('response_time')} 秒")
                                print(f"📄 内容プレビュー:")
                                
                                content = parsed.get('content', '')
                                preview = content[:300] + "..." if len(content) > 300 else content
                                print(f"「{preview}」")
                            else:
                                print(f"❌ エラー: {parsed.get('error')}")
                                
                        except json.JSONDecodeError:
                            print("JSON解析に失敗しました")
                            print(content_text)
                    else:
                        print("結果の取得に失敗しました")
                        
                except Exception as e:
                    print(f"❌ get_file_content テスト失敗: {e}")
                
                print()
        
        print("\n🎯 テスト完了")
        
    except ImportError as e:
        print("❌ FastMCPライブラリがインストールされていません")
        print("pip install fastmcp を実行してください")
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")
        if debug:
            logger.exception("詳細エラー情報:")

def main():
    """メイン関数"""
    parser = argparse.ArgumentParser(
        description="FastMCP Hybrid Search Server テストクライアント"
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="MCPサーバーのホスト (デフォルト: 127.0.0.1)"
    )

    parser.add_argument(
        "--port",
        default="8000",
        help="MCPサーバーのポート (デフォルト: 8000)"
    )
    
    parser.add_argument(
        "--query", "-q",
        default="Python ファイル処理",
        help="テスト検索クエリ (デフォルト: 'Python ファイル処理')"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["hybrid", "bm25", "vector"],
        default="hybrid",
        help="検索モード (デフォルト: hybrid)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="デバッグモードで実行"
    )
    
    parser.add_argument(
        "--file-path", "-f",
        help="get_file_contentツールでテストするファイルパス"
    )
    
    args = parser.parse_args()
    
    try:
        asyncio.run(test_fastmcp_server(
            host=args.host,
            port=args.port,
            query=args.query,
            mode=args.mode,
            debug=args.debug,
            test_file_path=args.file_path
        ))
    except KeyboardInterrupt:
        print("\n⏹️ テスト中断されました")
    except Exception as e:
        print(f"❌ テスト実行エラー: {e}")

if __name__ == "__main__":
    main() 