#!/usr/bin/env python3
"""
报告导出工具
支持将分析结果导出为多种格式
"""

import streamlit as st
import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import base64

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('web')

# 导入MongoDB报告管理器
try:
    from web.utils.mongodb_report_manager import mongodb_report_manager
    MONGODB_REPORT_AVAILABLE = True
except ImportError:
    MONGODB_REPORT_AVAILABLE = False
    mongodb_report_manager = None

# 配置日志 - 确保输出到stdout以便Docker logs可见
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到stdout
    ]
)
logger = logging.getLogger(__name__)

# 导入Docker适配器
try:
    from .docker_pdf_adapter import (
        is_docker_environment,
        get_docker_pdf_extra_args,
        setup_xvfb_display,
        get_docker_status_info
    )
    DOCKER_ADAPTER_AVAILABLE = True
except ImportError:
    DOCKER_ADAPTER_AVAILABLE = False
    logger.warning(f"⚠️ Docker适配器不可用")

# 导入导出相关库
try:
    import markdown
    import re
    import tempfile
    import os
    from pathlib import Path

    # 导入pypandoc（用于markdown转docx和pdf）
    import pypandoc

    # 检查pandoc是否可用，如果不可用则尝试下载
    try:
        pypandoc.get_pandoc_version()
        PANDOC_AVAILABLE = True
    except OSError:
        logger.warning(f"⚠️ 未找到pandoc，正在尝试自动下载...")
        try:
            pypandoc.download_pandoc()
            PANDOC_AVAILABLE = True
            logger.info(f"✅ pandoc下载成功！")
        except Exception as download_error:
            logger.error(f"❌ pandoc下载失败: {download_error}")
            PANDOC_AVAILABLE = False

    EXPORT_AVAILABLE = True

except ImportError as e:
    EXPORT_AVAILABLE = False
    PANDOC_AVAILABLE = False
    logger.info(f"导出功能依赖包缺失: {e}")
    logger.info(f"请安装: pip install pypandoc markdown")


class ReportExporter:
    """报告导出器"""

    def __init__(self):
        self.export_available = EXPORT_AVAILABLE
        self.pandoc_available = PANDOC_AVAILABLE
        self.is_docker = DOCKER_ADAPTER_AVAILABLE and is_docker_environment()

        # 记录初始化状态
        logger.info(f"📋 ReportExporter初始化:")
        logger.info(f"  - export_available: {self.export_available}")
        logger.info(f"  - pandoc_available: {self.pandoc_available}")
        logger.info(f"  - is_docker: {self.is_docker}")
        logger.info(f"  - docker_adapter_available: {DOCKER_ADAPTER_AVAILABLE}")

        # Docker环境初始化
        if self.is_docker:
            logger.info("🐳 检测到Docker环境，初始化PDF支持...")
            logger.info(f"🐳 检测到Docker环境，初始化PDF支持...")
            setup_xvfb_display()
    
    def _clean_text_for_markdown(self, text: str) -> str:
        """清理文本中可能导致YAML解析问题的字符"""
        if not text:
            return "N/A"

        # 转换为字符串并清理特殊字符
        text = str(text)

        # 移除可能导致YAML解析问题的字符
        text = text.replace('&', '&amp;')  # HTML转义
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        text = text.replace('"', '&quot;')
        text = text.replace("'", '&#39;')

        # 移除可能的YAML特殊字符
        text = text.replace('---', '—')  # 替换三个连字符
        text = text.replace('...', '…')  # 替换三个点

        return text

    def _clean_markdown_for_pandoc(self, content: str) -> str:
        """清理Markdown内容避免pandoc YAML解析问题"""
        if not content:
            return ""

        # 确保内容不以可能被误认为YAML的字符开头
        content = content.strip()

        # 如果第一行看起来像YAML分隔符，添加空行
        lines = content.split('\n')
        if lines and (lines[0].startswith('---') or lines[0].startswith('...')):
            content = '\n' + content

        # 替换可能导致YAML解析问题的字符序列，但保护表格分隔符
        # 先保护表格分隔符
        content = content.replace('|------|------|', '|TABLESEP|TABLESEP|')
        content = content.replace('|------|', '|TABLESEP|')

        # 然后替换其他的三连字符
        content = content.replace('---', '—')  # 替换三个连字符
        content = content.replace('...', '…')  # 替换三个点

        # 恢复表格分隔符
        content = content.replace('|TABLESEP|TABLESEP|', '|------|------|')
        content = content.replace('|TABLESEP|', '|------|')

        # 清理特殊引号
        content = content.replace('"', '"')  # 左双引号
        content = content.replace('"', '"')  # 右双引号
        content = content.replace(''', "'")  # 左单引号
        content = content.replace(''', "'")  # 右单引号

        # 确保内容以标准Markdown标题开始
        if not content.startswith('#'):
            content = '# 分析报告\n\n' + content

        return content

    def generate_markdown_report(self, results: Dict[str, Any]) -> str:
        """生成Markdown格式的报告"""

        stock_symbol = self._clean_text_for_markdown(results.get('stock_symbol', 'N/A'))
        decision = results.get('decision', {})
        state = results.get('state', {})
        is_demo = results.get('is_demo', False)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 清理关键数据
        action = self._clean_text_for_markdown(decision.get('action', 'N/A')).upper()
        target_price = self._clean_text_for_markdown(decision.get('target_price', 'N/A'))
        reasoning = self._clean_text_for_markdown(decision.get('reasoning', '暂无分析推理'))

        # 构建Markdown内容
        md_content = f"""# {stock_symbol} 股票分析报告

**生成时间**: {timestamp}
**分析状态**: {'演示模式' if is_demo else '正式分析'}

## 🎯 投资决策摘要

| 指标 | 数值 |
|------|------|
| **投资建议** | {action} |
| **置信度** | {decision.get('confidence', 0):.1%} |
| **风险评分** | {decision.get('risk_score', 0):.1%} |
| **目标价位** | {target_price} |

### 分析推理
{reasoning}

---

## 📋 分析配置信息

- **LLM提供商**: {results.get('llm_provider', 'N/A')}
- **AI模型**: {results.get('llm_model', 'N/A')}
- **分析师数量**: {len(results.get('analysts', []))}个
- **研究深度**: {results.get('research_depth', 'N/A')}

### 参与分析师
{', '.join(results.get('analysts', []))}

---

## 📊 详细分析报告

"""
        
        # 添加各个分析模块的内容 - 与CLI端保持一致的完整结构
        analysis_modules = [
            ('market_report', '📈 市场技术分析', '技术指标、价格趋势、支撑阻力位分析'),
            ('fundamentals_report', '💰 基本面分析', '财务数据、估值水平、盈利能力分析'),
            ('sentiment_report', '💭 市场情绪分析', '投资者情绪、社交媒体情绪指标'),
            ('news_report', '📰 新闻事件分析', '相关新闻事件、市场动态影响分析'),
            ('risk_assessment', '⚠️ 风险评估', '风险因素识别、风险等级评估'),
            ('investment_plan', '📋 投资建议', '具体投资策略、仓位管理建议')
        ]
        
        for key, title, description in analysis_modules:
            md_content += f"\n### {title}\n\n"
            md_content += f"*{description}*\n\n"
            
            if key in state and state[key]:
                content = state[key]
                if isinstance(content, str):
                    md_content += f"{content}\n\n"
                elif isinstance(content, dict):
                    for sub_key, sub_value in content.items():
                        md_content += f"#### {sub_key.replace('_', ' ').title()}\n\n"
                        md_content += f"{sub_value}\n\n"
                else:
                    md_content += f"{content}\n\n"
            else:
                md_content += "暂无数据\n\n"

        # 添加团队决策报告部分 - 与CLI端保持一致
        md_content = self._add_team_decision_reports(md_content, state)

        # 添加风险提示
        md_content += f"""
---

## ⚠️ 重要风险提示

**投资风险提示**:
- **仅供参考**: 本分析结果仅供参考，不构成投资建议
- **投资风险**: 股票投资有风险，可能导致本金损失
- **理性决策**: 请结合多方信息进行理性投资决策
- **专业咨询**: 重大投资决策建议咨询专业财务顾问
- **自担风险**: 投资决策及其后果由投资者自行承担

---
*报告生成时间: {timestamp}*
"""
        
        return md_content

    def _add_team_decision_reports(self, md_content: str, state: Dict[str, Any]) -> str:
        """添加团队决策报告部分，与CLI端保持一致"""

        # II. 研究团队决策报告
        if 'investment_debate_state' in state and state['investment_debate_state']:
            md_content += "\n---\n\n## 🔬 研究团队决策\n\n"
            md_content += "*多头/空头研究员辩论分析，研究经理综合决策*\n\n"

            debate_state = state['investment_debate_state']

            # 多头研究员分析
            if debate_state.get('bull_history'):
                md_content += "### 📈 多头研究员分析\n\n"
                md_content += f"{self._clean_text_for_markdown(debate_state['bull_history'])}\n\n"

            # 空头研究员分析
            if debate_state.get('bear_history'):
                md_content += "### 📉 空头研究员分析\n\n"
                md_content += f"{self._clean_text_for_markdown(debate_state['bear_history'])}\n\n"

            # 研究经理决策
            if debate_state.get('judge_decision'):
                md_content += "### 🎯 研究经理综合决策\n\n"
                md_content += f"{self._clean_text_for_markdown(debate_state['judge_decision'])}\n\n"

        # III. 交易团队计划
        if 'trader_investment_plan' in state and state['trader_investment_plan']:
            md_content += "\n---\n\n## 💼 交易团队计划\n\n"
            md_content += "*专业交易员制定的具体交易执行计划*\n\n"
            md_content += f"{self._clean_text_for_markdown(state['trader_investment_plan'])}\n\n"

        # IV. 风险管理团队决策
        if 'risk_debate_state' in state and state['risk_debate_state']:
            md_content += "\n---\n\n## ⚖️ 风险管理团队决策\n\n"
            md_content += "*激进/保守/中性分析师风险评估，投资组合经理最终决策*\n\n"

            risk_state = state['risk_debate_state']

            # 激进分析师
            if risk_state.get('risky_history'):
                md_content += "### 🚀 激进分析师评估\n\n"
                md_content += f"{self._clean_text_for_markdown(risk_state['risky_history'])}\n\n"

            # 保守分析师
            if risk_state.get('safe_history'):
                md_content += "### 🛡️ 保守分析师评估\n\n"
                md_content += f"{self._clean_text_for_markdown(risk_state['safe_history'])}\n\n"

            # 中性分析师
            if risk_state.get('neutral_history'):
                md_content += "### ⚖️ 中性分析师评估\n\n"
                md_content += f"{self._clean_text_for_markdown(risk_state['neutral_history'])}\n\n"

            # 投资组合经理决策
            if risk_state.get('judge_decision'):
                md_content += "### 🎯 投资组合经理最终决策\n\n"
                md_content += f"{self._clean_text_for_markdown(risk_state['judge_decision'])}\n\n"

        # V. 最终交易决策
        if 'final_trade_decision' in state and state['final_trade_decision']:
            md_content += "\n---\n\n## 🎯 最终交易决策\n\n"
            md_content += "*综合所有团队分析后的最终投资决策*\n\n"
            md_content += f"{self._clean_text_for_markdown(state['final_trade_decision'])}\n\n"

        return md_content

    def _format_team_decision_content(self, content: Dict[str, Any], module_key: str) -> str:
        """格式化团队决策内容"""
        formatted_content = ""

        if module_key == 'investment_debate_state':
            # 研究团队决策格式化
            if content.get('bull_history'):
                formatted_content += "## 📈 多头研究员分析\n\n"
                formatted_content += f"{content['bull_history']}\n\n"

            if content.get('bear_history'):
                formatted_content += "## 📉 空头研究员分析\n\n"
                formatted_content += f"{content['bear_history']}\n\n"

            if content.get('judge_decision'):
                formatted_content += "## 🎯 研究经理综合决策\n\n"
                formatted_content += f"{content['judge_decision']}\n\n"

        elif module_key == 'risk_debate_state':
            # 风险管理团队决策格式化
            if content.get('risky_history'):
                formatted_content += "## 🚀 激进分析师评估\n\n"
                formatted_content += f"{content['risky_history']}\n\n"

            if content.get('safe_history'):
                formatted_content += "## 🛡️ 保守分析师评估\n\n"
                formatted_content += f"{content['safe_history']}\n\n"

            if content.get('neutral_history'):
                formatted_content += "## ⚖️ 中性分析师评估\n\n"
                formatted_content += f"{content['neutral_history']}\n\n"

            if content.get('judge_decision'):
                formatted_content += "## 🎯 投资组合经理最终决策\n\n"
                formatted_content += f"{content['judge_decision']}\n\n"

        return formatted_content

    def generate_docx_report(self, results: Dict[str, Any]) -> bytes:
        """生成Word文档格式的报告"""

        logger.info("📄 开始生成Word文档...")

        if not self.pandoc_available:
            logger.error("❌ Pandoc不可用")
            raise Exception("Pandoc不可用，无法生成Word文档。请安装pandoc或使用Markdown格式导出。")

        # 首先生成markdown内容
        logger.info("📝 生成Markdown内容...")
        md_content = self.generate_markdown_report(results)
        logger.info(f"✅ Markdown内容生成完成，长度: {len(md_content)} 字符")

        try:
            logger.info("📁 创建临时文件用于docx输出...")
            # 创建临时文件用于docx输出
            with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp_file:
                output_file = tmp_file.name
            logger.info(f"📁 临时文件路径: {output_file}")

            # 使用强制禁用YAML的参数
            extra_args = ['--from=markdown-yaml_metadata_block']  # 禁用YAML解析
            logger.info(f"🔧 pypandoc参数: {extra_args} (禁用YAML解析)")

            logger.info("🔄 使用pypandoc将markdown转换为docx...")

            # 调试：保存实际的Markdown内容
            debug_file = '/app/debug_markdown.md'
            try:
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(md_content)
                logger.info(f"🔍 实际Markdown内容已保存到: {debug_file}")
                logger.info(f"📊 内容长度: {len(md_content)} 字符")

                # 显示前几行内容
                lines = md_content.split('\n')[:5]
                logger.info("🔍 前5行内容:")
                for i, line in enumerate(lines, 1):
                    logger.info(f"  {i}: {repr(line)}")
            except Exception as e:
                logger.error(f"保存调试文件失败: {e}")

            # 清理内容避免YAML解析问题
            cleaned_content = self._clean_markdown_for_pandoc(md_content)
            logger.info(f"🧹 内容清理完成，清理后长度: {len(cleaned_content)} 字符")

            # 使用测试成功的参数进行转换
            pypandoc.convert_text(
                cleaned_content,
                'docx',
                format='markdown',  # 基础markdown格式
                outputfile=output_file,
                extra_args=extra_args
            )
            logger.info("✅ pypandoc转换完成")

            logger.info("📖 读取生成的docx文件...")
            # 读取生成的docx文件
            with open(output_file, 'rb') as f:
                docx_content = f.read()
            logger.info(f"✅ 文件读取完成，大小: {len(docx_content)} 字节")

            logger.info("🗑️ 清理临时文件...")
            # 清理临时文件
            os.unlink(output_file)
            logger.info("✅ 临时文件清理完成")

            return docx_content
        except Exception as e:
            logger.error(f"❌ Word文档生成失败: {e}", exc_info=True)
            raise Exception(f"生成Word文档失败: {e}")
    
    
    def generate_pdf_report(self, results: Dict[str, Any]) -> bytes:
        """生成PDF格式的报告"""

        logger.info("📊 开始生成PDF文档...")

        if not self.pandoc_available:
            logger.error("❌ Pandoc不可用")
            raise Exception("Pandoc不可用，无法生成PDF文档。请安装pandoc或使用Markdown格式导出。")

        # 首先生成markdown内容
        logger.info("📝 生成Markdown内容...")
        md_content = self.generate_markdown_report(results)
        logger.info(f"✅ Markdown内容生成完成，长度: {len(md_content)} 字符")

        # 简化的PDF引擎列表，优先使用最可能成功的
        pdf_engines = [
            ('wkhtmltopdf', 'HTML转PDF引擎，推荐安装'),
            ('weasyprint', '现代HTML转PDF引擎'),
            (None, '使用pandoc默认引擎')  # 不指定引擎，让pandoc自己选择
        ]

        last_error = None

        for engine_info in pdf_engines:
            engine, description = engine_info
            try:
                # 创建临时文件用于PDF输出
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    output_file = tmp_file.name

                # 使用禁用YAML解析的参数（与Word导出一致）
                extra_args = ['--from=markdown-yaml_metadata_block']

                # 如果指定了引擎，添加引擎参数
                if engine:
                    extra_args.append(f'--pdf-engine={engine}')
                    logger.info(f"🔧 使用PDF引擎: {engine}")
                else:
                    logger.info(f"🔧 使用默认PDF引擎")

                logger.info(f"🔧 PDF参数: {extra_args}")

                # 清理内容避免YAML解析问题（与Word导出一致）
                cleaned_content = self._clean_markdown_for_pandoc(md_content)

                # 使用pypandoc将markdown转换为PDF - 禁用YAML解析
                pypandoc.convert_text(
                    cleaned_content,
                    'pdf',
                    format='markdown',  # 基础markdown格式
                    outputfile=output_file,
                    extra_args=extra_args
                )

                # 检查文件是否生成且有内容
                if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
                    # 读取生成的PDF文件
                    with open(output_file, 'rb') as f:
                        pdf_content = f.read()

                    # 清理临时文件
                    os.unlink(output_file)

                    logger.info(f"✅ PDF生成成功，使用引擎: {engine or '默认'}")
                    return pdf_content
                else:
                    raise Exception("PDF文件生成失败或为空")

            except Exception as e:
                last_error = str(e)
                logger.error(f"PDF引擎 {engine or '默认'} 失败: {e}")

                # 清理可能存在的临时文件
                try:
                    if 'output_file' in locals() and os.path.exists(output_file):
                        os.unlink(output_file)
                except:
                    pass

                continue

        # 如果所有引擎都失败，提供详细的错误信息和解决方案
        error_msg = f"""PDF生成失败，最后错误: {last_error}

可能的解决方案:
1. 安装wkhtmltopdf (推荐):
   Windows: choco install wkhtmltopdf
   macOS: brew install wkhtmltopdf
   Linux: sudo apt-get install wkhtmltopdf

2. 安装LaTeX:
   Windows: choco install miktex
   macOS: brew install mactex
   Linux: sudo apt-get install texlive-full

3. 使用Markdown或Word格式导出作为替代方案
"""
        raise Exception(error_msg)
    
    def export_report(self, results: Dict[str, Any], format_type: str) -> Optional[bytes]:
        """导出报告为指定格式"""

        logger.info(f"🚀 开始导出报告: format={format_type}")
        logger.info(f"📊 导出状态检查:")
        logger.info(f"  - export_available: {self.export_available}")
        logger.info(f"  - pandoc_available: {self.pandoc_available}")
        logger.info(f"  - is_docker: {self.is_docker}")

        if not self.export_available:
            logger.error("❌ 导出功能不可用")
            st.error("❌ 导出功能不可用，请安装必要的依赖包")
            return None

        try:
            logger.info(f"🔄 开始生成{format_type}格式报告...")

            if format_type == 'markdown':
                logger.info("📝 生成Markdown报告...")
                content = self.generate_markdown_report(results)
                logger.info(f"✅ Markdown报告生成成功，长度: {len(content)} 字符")
                return content.encode('utf-8')

            elif format_type == 'docx':
                logger.info("📄 生成Word文档...")
                if not self.pandoc_available:
                    logger.error("❌ pandoc不可用，无法生成Word文档")
                    st.error("❌ pandoc不可用，无法生成Word文档")
                    return None
                content = self.generate_docx_report(results)
                logger.info(f"✅ Word文档生成成功，大小: {len(content)} 字节")
                return content

            elif format_type == 'pdf':
                logger.info("📊 生成PDF文档...")
                if not self.pandoc_available:
                    logger.error("❌ pandoc不可用，无法生成PDF文档")
                    st.error("❌ pandoc不可用，无法生成PDF文档")
                    return None
                content = self.generate_pdf_report(results)
                logger.info(f"✅ PDF文档生成成功，大小: {len(content)} 字节")
                return content

            else:
                logger.error(f"❌ 不支持的导出格式: {format_type}")
                st.error(f"❌ 不支持的导出格式: {format_type}")
                return None

        except Exception as e:
            logger.error(f"❌ 导出失败: {str(e)}", exc_info=True)
            st.error(f"❌ 导出失败: {str(e)}")
            return None


# 创建全局导出器实例
report_exporter = ReportExporter()


def _format_team_decision_content(content: Dict[str, Any], module_key: str) -> str:
    """格式化团队决策内容（独立函数版本）"""
    formatted_content = ""

    if module_key == 'investment_debate_state':
        # 研究团队决策格式化
        if content.get('bull_history'):
            formatted_content += "## 📈 多头研究员分析\n\n"
            formatted_content += f"{content['bull_history']}\n\n"

        if content.get('bear_history'):
            formatted_content += "## 📉 空头研究员分析\n\n"
            formatted_content += f"{content['bear_history']}\n\n"

        if content.get('judge_decision'):
            formatted_content += "## 🎯 研究经理综合决策\n\n"
            formatted_content += f"{content['judge_decision']}\n\n"

    elif module_key == 'risk_debate_state':
        # 风险管理团队决策格式化
        if content.get('risky_history'):
            formatted_content += "## 🚀 激进分析师评估\n\n"
            formatted_content += f"{content['risky_history']}\n\n"

        if content.get('safe_history'):
            formatted_content += "## 🛡️ 保守分析师评估\n\n"
            formatted_content += f"{content['safe_history']}\n\n"

        if content.get('neutral_history'):
            formatted_content += "## ⚖️ 中性分析师评估\n\n"
            formatted_content += f"{content['neutral_history']}\n\n"

        if content.get('judge_decision'):
            formatted_content += "## 🎯 投资组合经理最终决策\n\n"
            formatted_content += f"{content['judge_decision']}\n\n"

    return formatted_content


def save_modular_reports_to_results_dir(results: Dict[str, Any], stock_symbol: str, analysis_id: Optional[str] = None) -> Dict[str, str]:
    """保存分模块报告到results目录（CLI版本格式）"""
    try:
        import os
        from pathlib import Path

        # 获取项目根目录
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent

        # 获取results目录配置
        results_dir_env = os.getenv("TRADINGAGENTS_RESULTS_DIR")
        if results_dir_env:
            if not os.path.isabs(results_dir_env):
                results_dir = project_root / results_dir_env
            else:
                results_dir = Path(results_dir_env)
        else:
            results_dir = project_root / "results"

        # 创建股票专用目录
        analysis_date = datetime.now().strftime('%Y-%m-%d')
        stock_dir = results_dir / stock_symbol / analysis_date
        reports_dir = stock_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        # 创建message_tool.log文件
        log_file = stock_dir / "message_tool.log"
        log_file.touch(exist_ok=True)

        state = results.get('state', {})
        saved_files = {}

        # 定义报告模块映射（与CLI版本保持一致）
        report_modules = {
            'market_report': {
                'filename': 'market_report.md',
                'title': f'{stock_symbol} 股票技术分析报告',
                'state_key': 'market_report'
            },
            'sentiment_report': {
                'filename': 'sentiment_report.md',
                'title': f'{stock_symbol} 市场情绪分析报告',
                'state_key': 'sentiment_report'
            },
            'news_report': {
                'filename': 'news_report.md',
                'title': f'{stock_symbol} 新闻事件分析报告',
                'state_key': 'news_report'
            },
            'fundamentals_report': {
                'filename': 'fundamentals_report.md',
                'title': f'{stock_symbol} 基本面分析报告',
                'state_key': 'fundamentals_report'
            },
            'investment_plan': {
                'filename': 'investment_plan.md',
                'title': f'{stock_symbol} 投资决策报告',
                'state_key': 'investment_plan'
            },
            'trader_investment_plan': {
                'filename': 'trader_investment_plan.md',
                'title': f'{stock_symbol} 交易计划报告',
                'state_key': 'trader_investment_plan'
            },
            'final_trade_decision': {
                'filename': 'final_trade_decision.md',
                'title': f'{stock_symbol} 最终投资决策',
                'state_key': 'final_trade_decision'
            },
            # 添加团队决策报告模块
            'investment_debate_state': {
                'filename': 'research_team_decision.md',
                'title': f'{stock_symbol} 研究团队决策报告',
                'state_key': 'investment_debate_state'
            },
            'risk_debate_state': {
                'filename': 'risk_management_decision.md',
                'title': f'{stock_symbol} 风险管理团队决策报告',
                'state_key': 'risk_debate_state'
            }
        }

        # 生成各个模块的报告文件
        for module_key, module_info in report_modules.items():
            content = state.get(module_info['state_key'])

            if content:
                # 生成模块报告内容
                if isinstance(content, str):
                    # 检查内容是否已经包含标题，避免重复添加
                    if content.strip().startswith('#'):
                        report_content = content
                    else:
                        report_content = f"# {module_info['title']}\n\n{content}"
                elif isinstance(content, dict):
                    report_content = f"# {module_info['title']}\n\n"
                    # 特殊处理团队决策报告的字典结构
                    if module_key in ['investment_debate_state', 'risk_debate_state']:
                        report_content += _format_team_decision_content(content, module_key)
                    else:
                        for sub_key, sub_value in content.items():
                            report_content += f"## {sub_key.replace('_', ' ').title()}\n\n{sub_value}\n\n"
                else:
                    report_content = f"# {module_info['title']}\n\n{str(content)}"

                # 保存文件
                file_path = reports_dir / module_info['filename']
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)

                saved_files[module_key] = str(file_path)
                logger.info(f"✅ 保存模块报告: {file_path}")

        # 如果有决策信息，也保存最终决策报告
        decision = results.get('decision', {})
        if decision:
            decision_content = f"# {stock_symbol} 最终投资决策\n\n"

            if isinstance(decision, dict):
                decision_content += f"## 投资建议\n\n"
                decision_content += f"**行动**: {decision.get('action', 'N/A')}\n\n"
                decision_content += f"**置信度**: {decision.get('confidence', 0):.1%}\n\n"
                decision_content += f"**风险评分**: {decision.get('risk_score', 0):.1%}\n\n"
                decision_content += f"**目标价位**: {decision.get('target_price', 'N/A')}\n\n"
                decision_content += f"## 分析推理\n\n{decision.get('reasoning', '暂无分析推理')}\n\n"
            else:
                decision_content += f"{str(decision)}\n\n"

            decision_file = reports_dir / "final_trade_decision.md"
            with open(decision_file, 'w', encoding='utf-8') as f:
                f.write(decision_content)

            saved_files['final_trade_decision'] = str(decision_file)
            logger.info(f"✅ 保存最终决策: {decision_file}")

        # 保存分析元数据文件，包含研究深度等信息
        metadata = {
            'stock_symbol': stock_symbol,
            'analysis_date': analysis_date,
            'timestamp': datetime.now().isoformat(),
            'research_depth': results.get('research_depth', 1),
            'analysts': results.get('analysts', []),
            'status': 'completed',
            'reports_count': len(saved_files),
            'report_types': list(saved_files.keys())
        }

        metadata_file = reports_dir.parent / "analysis_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"✅ 保存分析元数据: {metadata_file}")
        logger.info(f"✅ 分模块报告保存完成，共保存 {len(saved_files)} 个文件")
        logger.info(f"📁 保存目录: {os.path.normpath(str(reports_dir))}")

        # 同时保存到MongoDB
        logger.info(f"🔍 [MongoDB调试] 开始MongoDB保存流程")
        logger.info(f"🔍 [MongoDB调试] MONGODB_REPORT_AVAILABLE: {MONGODB_REPORT_AVAILABLE}")
        logger.info(f"🔍 [MongoDB调试] mongodb_report_manager存在: {mongodb_report_manager is not None}")

        if MONGODB_REPORT_AVAILABLE and mongodb_report_manager:
            logger.info(f"🔍 [MongoDB调试] MongoDB管理器连接状态: {mongodb_report_manager.connected}")
            try:
                # 收集所有报告内容
                reports_content = {}

                # 复用/获取analysis_id
                if analysis_id is None:
                    analysis_id = results.get('analysis_id')
                logger.info(f"🔍 [MongoDB调试] 使用analysis_id: {analysis_id}")

                logger.info(f"🔍 [MongoDB调试] 开始读取 {len(saved_files)} 个报告文件")
                # 读取已保存的文件内容
                for module_key, file_path in saved_files.items():
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            reports_content[module_key] = content
                            logger.info(f"🔍 [MongoDB调试] 成功读取 {module_key}: {len(content)} 字符")
                    except Exception as e:
                        logger.warning(f"⚠️ 读取报告文件失败 {file_path}: {e}")

                # 保存到MongoDB
                if reports_content:
                    logger.info(f"🔍 [MongoDB调试] 准备保存到MongoDB，报告数量: {len(reports_content)}")
                    logger.info(f"🔍 [MongoDB调试] 报告类型: {list(reports_content.keys())}")

                    success = mongodb_report_manager.save_analysis_report(
                        stock_symbol=stock_symbol,
                        analysis_results=results,
                        reports=reports_content,
                        analysis_id=analysis_id
                    )

                    if success:
                        logger.info(f"✅ 分析报告已同时保存到MongoDB")
                    else:
                        logger.warning(f"⚠️ MongoDB保存失败，但文件保存成功")
                else:
                    logger.warning(f"⚠️ 没有报告内容可保存到MongoDB")

            except Exception as e:
                logger.error(f"❌ MongoDB保存过程出错: {e}")
                import traceback
                logger.error(f"❌ MongoDB保存详细错误: {traceback.format_exc()}")
                # 不影响文件保存的成功返回
        else:
            logger.warning(f"⚠️ MongoDB保存跳过 - AVAILABLE: {MONGODB_REPORT_AVAILABLE}, Manager: {mongodb_report_manager is not None}")

        return saved_files

    except Exception as e:
        logger.error(f"❌ 保存分模块报告失败: {e}")
        import traceback
        logger.error(f"❌ 详细错误: {traceback.format_exc()}")
        return {}


def save_report_to_results_dir(content: bytes, filename: str, stock_symbol: str) -> str:
    """保存报告到results目录"""
    try:
        import os
        from pathlib import Path

        # 获取项目根目录（Web应用在web/子目录中运行）
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent  # web/utils/report_exporter.py -> 项目根目录

        # 获取results目录配置
        results_dir_env = os.getenv("TRADINGAGENTS_RESULTS_DIR")
        if results_dir_env:
            # 如果环境变量是相对路径，相对于项目根目录解析
            if not os.path.isabs(results_dir_env):
                results_dir = project_root / results_dir_env
            else:
                results_dir = Path(results_dir_env)
        else:
            # 默认使用项目根目录下的results
            results_dir = project_root / "results"

        # 创建股票专用目录
        analysis_date = datetime.now().strftime('%Y-%m-%d')
        stock_dir = results_dir / stock_symbol / analysis_date / "reports"
        stock_dir.mkdir(parents=True, exist_ok=True)

        # 保存文件
        file_path = stock_dir / filename
        with open(file_path, 'wb') as f:
            f.write(content)

        logger.info(f"✅ 报告已保存到: {file_path}")
        logger.info(f"📁 项目根目录: {project_root}")
        logger.info(f"📁 Results目录: {results_dir}")
        logger.info(f"📁 环境变量TRADINGAGENTS_RESULTS_DIR: {results_dir_env}")

        return str(file_path)

    except Exception as e:
        logger.error(f"❌ 保存报告到results目录失败: {e}")
        import traceback
        logger.error(f"❌ 详细错误: {traceback.format_exc()}")
        return ""


def render_export_buttons(results: Dict[str, Any]):
    """渲染导出按钮"""

    if not results:
        return

    st.markdown("---")
    st.subheader("📤 导出报告")

    # 检查导出功能是否可用
    if not report_exporter.export_available:
        st.warning("⚠️ 导出功能需要安装额外依赖包")
        st.code("pip install pypandoc markdown")
        return

    # 检查pandoc是否可用
    if not report_exporter.pandoc_available:
        st.warning("⚠️ Word和PDF导出需要pandoc工具")
        st.info("💡 您仍可以使用Markdown格式导出")

    # 显示Docker环境状态
    if report_exporter.is_docker:
        if DOCKER_ADAPTER_AVAILABLE:
            docker_status = get_docker_status_info()
            if docker_status['dependencies_ok'] and docker_status['pdf_test_ok']:
                st.success("🐳 Docker环境PDF支持已启用")
            else:
                st.warning(f"🐳 Docker环境PDF支持异常: {docker_status['dependency_message']}")
        else:
            st.warning("🐳 Docker环境检测到，但适配器不可用")

        with st.expander("📖 如何安装pandoc"):
            st.markdown("""
            **Windows用户:**
            ```bash
            # 使用Chocolatey (推荐)
            choco install pandoc

            # 或下载安装包
            # https://github.com/jgm/pandoc/releases
            ```

            **或者使用Python自动下载:**
            ```python
            import pypandoc

            pypandoc.download_pandoc()
            ```
            """)

        # 在Docker环境下，即使pandoc有问题也显示所有按钮，让用户尝试
        pass
    
    # 生成文件名
    stock_symbol = results.get('stock_symbol', 'analysis')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 导出 Markdown", help="导出为Markdown格式"):
            logger.info(f"🖱️ [EXPORT] 用户点击Markdown导出按钮 - 股票: {stock_symbol}")
            logger.info(f"🖱️ 用户点击Markdown导出按钮 - 股票: {stock_symbol}")
            # 1. 保存分模块报告（CLI格式）
            logger.info("📁 开始保存分模块报告（CLI格式）...")
            modular_files = save_modular_reports_to_results_dir(results, stock_symbol)

            # 2. 生成汇总报告（下载用）
            content = report_exporter.export_report(results, 'markdown')
            if content:
                filename = f"{stock_symbol}_analysis_{timestamp}.md"
                logger.info(f"✅ [EXPORT] Markdown导出成功，文件名: {filename}")
                logger.info(f"✅ Markdown导出成功，文件名: {filename}")

                # 3. 保存汇总报告到results目录
                saved_path = save_report_to_results_dir(content, filename, stock_symbol)

                # 4. 显示保存结果
                if modular_files and saved_path:
                    st.success(f"✅ 已保存 {len(modular_files)} 个分模块报告 + 1个汇总报告")
                    with st.expander("📁 查看保存的文件"):
                        st.write("**分模块报告:**")
                        for module, path in modular_files.items():
                            st.write(f"- {module}: `{path}`")
                        st.write("**汇总报告:**")
                        st.write(f"- 汇总报告: `{saved_path}`")
                elif saved_path:
                    st.success(f"✅ 汇总报告已保存到: {saved_path}")

                st.download_button(
                    label="📥 下载 Markdown",
                    data=content,
                    file_name=filename,
                    mime="text/markdown"
                )
            else:
                logger.error(f"❌ [EXPORT] Markdown导出失败，content为空")
                logger.error("❌ Markdown导出失败，content为空")
    
    with col2:
        if st.button("📝 导出 Word", help="导出为Word文档格式"):
            logger.info(f"🖱️ [EXPORT] 用户点击Word导出按钮 - 股票: {stock_symbol}")
            logger.info(f"🖱️ 用户点击Word导出按钮 - 股票: {stock_symbol}")
            with st.spinner("正在生成Word文档，请稍候..."):
                try:
                    logger.info(f"🔄 [EXPORT] 开始Word导出流程...")
                    logger.info("🔄 开始Word导出流程...")

                    # 1. 保存分模块报告（CLI格式）
                    logger.info("📁 开始保存分模块报告（CLI格式）...")
                    modular_files = save_modular_reports_to_results_dir(results, stock_symbol)

                    # 2. 生成Word汇总报告
                    content = report_exporter.export_report(results, 'docx')
                    if content:
                        filename = f"{stock_symbol}_analysis_{timestamp}.docx"
                        logger.info(f"✅ [EXPORT] Word导出成功，文件名: {filename}, 大小: {len(content)} 字节")
                        logger.info(f"✅ Word导出成功，文件名: {filename}, 大小: {len(content)} 字节")

                        # 3. 保存Word汇总报告到results目录
                        saved_path = save_report_to_results_dir(content, filename, stock_symbol)

                        # 4. 显示保存结果
                        if modular_files and saved_path:
                            st.success(f"✅ 已保存 {len(modular_files)} 个分模块报告 + 1个Word汇总报告")
                            with st.expander("📁 查看保存的文件"):
                                st.write("**分模块报告:**")
                                for module, path in modular_files.items():
                                    st.write(f"- {module}: `{path}`")
                                st.write("**Word汇总报告:**")
                                st.write(f"- Word报告: `{saved_path}`")
                        elif saved_path:
                            st.success(f"✅ Word文档已保存到: {saved_path}")
                        else:
                            st.success("✅ Word文档生成成功！")

                        st.download_button(
                            label="📥 下载 Word",
                            data=content,
                            file_name=filename,
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                    else:
                        logger.error(f"❌ [EXPORT] Word导出失败，content为空")
                        logger.error("❌ Word导出失败，content为空")
                        st.error("❌ Word文档生成失败")
                except Exception as e:
                    logger.error(f"❌ [EXPORT] Word导出异常: {str(e)}")
                    logger.error(f"❌ Word导出异常: {str(e)}", exc_info=True)
                    st.error(f"❌ Word文档生成失败: {str(e)}")

                    # 显示详细错误信息
                    with st.expander("🔍 查看详细错误信息"):
                        st.text(str(e))

                    # 提供解决方案
                    with st.expander("💡 解决方案"):
                        st.markdown("""
                        **Word导出需要pandoc工具，请检查:**

                        1. **Docker环境**: 重新构建镜像确保包含pandoc
                        2. **本地环境**: 安装pandoc
                        ```bash
                        # Windows
                        choco install pandoc

                        # macOS
                        brew install pandoc

                        # Linux
                        sudo apt-get install pandoc
                        ```

                        3. **替代方案**: 使用Markdown格式导出
                        """)
    
    with col3:
        if st.button("📊 导出 PDF", help="导出为PDF格式 (需要额外工具)"):
            logger.info(f"🖱️ 用户点击PDF导出按钮 - 股票: {stock_symbol}")
            with st.spinner("正在生成PDF，请稍候..."):
                try:
                    logger.info("🔄 开始PDF导出流程...")

                    # 1. 保存分模块报告（CLI格式）
                    logger.info("📁 开始保存分模块报告（CLI格式）...")
                    modular_files = save_modular_reports_to_results_dir(results, stock_symbol)

                    # 2. 生成PDF汇总报告
                    content = report_exporter.export_report(results, 'pdf')
                    if content:
                        filename = f"{stock_symbol}_analysis_{timestamp}.pdf"
                        logger.info(f"✅ PDF导出成功，文件名: {filename}, 大小: {len(content)} 字节")

                        # 3. 保存PDF汇总报告到results目录
                        saved_path = save_report_to_results_dir(content, filename, stock_symbol)

                        # 4. 显示保存结果
                        if modular_files and saved_path:
                            st.success(f"✅ 已保存 {len(modular_files)} 个分模块报告 + 1个PDF汇总报告")
                            with st.expander("📁 查看保存的文件"):
                                st.write("**分模块报告:**")
                                for module, path in modular_files.items():
                                    st.write(f"- {module}: `{path}`")
                                st.write("**PDF汇总报告:**")
                                st.write(f"- PDF报告: `{saved_path}`")
                        elif saved_path:
                            st.success(f"✅ PDF已保存到: {saved_path}")
                        else:
                            st.success("✅ PDF生成成功！")

                        st.download_button(
                            label="📥 下载 PDF",
                            data=content,
                            file_name=filename,
                            mime="application/pdf"
                        )
                    else:
                        logger.error("❌ PDF导出失败，content为空")
                        st.error("❌ PDF生成失败")
                except Exception as e:
                    logger.error(f"❌ PDF导出异常: {str(e)}", exc_info=True)
                    st.error(f"❌ PDF生成失败")

                    # 显示详细错误信息
                    with st.expander("🔍 查看详细错误信息"):
                        st.text(str(e))

                    # 提供解决方案
                    with st.expander("💡 解决方案"):
                        st.markdown("""
                        **PDF导出需要额外的工具，请选择以下方案之一:**

                        **方案1: 安装wkhtmltopdf (推荐)**
                        ```bash
                        # Windows
                        choco install wkhtmltopdf

                        # macOS
                        brew install wkhtmltopdf

                        # Linux
                        sudo apt-get install wkhtmltopdf
                        ```

                        **方案2: 安装LaTeX**
                        ```bash
                        # Windows
                        choco install miktex

                        # macOS
                        brew install mactex

                        # Linux
                        sudo apt-get install texlive-full
                        ```

                        **方案3: 使用替代格式**
                        - 📄 Markdown格式 - 轻量级，兼容性好
                        - 📝 Word格式 - 适合进一步编辑
                        """)

                    # 建议使用其他格式
                    st.info("💡 建议：您可以先使用Markdown或Word格式导出，然后使用其他工具转换为PDF")


def save_analysis_report(stock_symbol: str, analysis_results: Dict[str, Any], 
                        report_content: str = None, analysis_id: Optional[str] = None) -> bool:
    """
    保存分析报告到MongoDB
    
    Args:
        stock_symbol: 股票代码
        analysis_results: 分析结果字典
        report_content: 报告内容（可选，如果不提供则自动生成）
    
    Returns:
        bool: 保存是否成功
    """
    try:
        if not MONGODB_REPORT_AVAILABLE or mongodb_report_manager is None:
            logger.warning("MongoDB报告管理器不可用，无法保存报告")
            return False
        
        # 如果没有提供报告内容，则生成Markdown报告
        if report_content is None:
            report_content = report_exporter.generate_markdown_report(analysis_results)
        
        # 调用MongoDB报告管理器保存报告
        # 将报告内容包装成字典格式
        reports_dict = {
            "markdown": report_content,
            "generated_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        success = mongodb_report_manager.save_analysis_report(
            stock_symbol=stock_symbol,
            analysis_results=analysis_results,
            reports=reports_dict,
            analysis_id=analysis_id or analysis_results.get('analysis_id')
        )
        
        if success:
            logger.info(f"✅ 分析报告已成功保存到MongoDB - 股票: {stock_symbol}")
        else:
            logger.error(f"❌ 分析报告保存到MongoDB失败 - 股票: {stock_symbol}")
        
        return success
        
    except Exception as e:
        logger.error(f"❌ 保存分析报告到MongoDB时发生异常 - 股票: {stock_symbol}, 错误: {str(e)}")
        return False
    
 