# 每日研究执行契约

## 状态在哪里

GitHub 远端的 [queue](../research/queue.json)、[progress](../research/progress.json)、[sources](../research/sources.json)、[evidence](../research/evidence.json) 是跨运行权威，不依赖上一聊天记忆。

研究 `done` 只代表该问题契约与证据完成，不能升级用户掌握或硬件实践。Pending/blocked 是真实研究状态，不是验证工具的失败。

## 分支、主分支同步与防重叠

每次在隔离 worktree 基于最新 `origin/main` 增量研究；若远端存在未合并历史研究分支，先合入当前分支以确保不丢数据。通过全部单测与结构验证后直接 `git push origin HEAD:main` 同步到主分支（禁止 force-push）。

通过 [research_lock 工具](../scripts/README.md) 在 common Git directory 建立持久锁，跨 worktree 共用。获取者使用当前会话的 opaque owner；发现已有锁时先核验 owner 的 App 实时状态。**不按时间/PID 自动抢锁**；确认持有者不再工作后才显式释放。所有结束路径释放自己的锁。

不同工作树/分支也不并发推送冲突变更。Push 遭拒后先 fetch 并核验 `origin/main` 远端变化，有界合入解决；禁止 force-push、不丢弃已有提交。

## 一次有界运行

1. 核验 owner 与远端状态（`origin/main` 及任何未合并历史研究分支），获取锁，读取台账与最近日报。
2. 从已满足全部硬先修的 queue 中选一个问题；优先 GPU/HBM/互连，不盲目追热点。
3. 运行来源监控。部分失败留明确信息并使用已读合法材料；不能宣称完成全面扫描。
4. 阅读正文/固定源码，完成一个实质问题；允许原创 CPU/offline 实验，GPU/云/API 另行批准。
5. 更新组件、来源/主张、问题状态、趋势与 `docs\daily\YYYY-MM-DD.md`。日期按北京时间 UTC+08:00；同日只增量更新一个文件。
6. 执行单元测试、结构验证、实验 `--check`，审阅 staged 文件及公开边界。
7. 普通提交并通过 `git push origin HEAD:main` 同步到主分支（禁止 force-push）；保留 pending/blocked 问题，记录下一步，释放锁。

公开仓库的提交在 push 后即刻可见。发布检查必须先于 push，不复制任何私人历史/第三方全文，禁止提交缓存、秘密、签名 URL、内部来源和真实账户/客户资料。

## 日报质量

每次至少推进一个可验证问题；日报链接到新增的实质证据，不以改日期当研究。当天没有可信新发布可以深化旧问题，明确“未发现经核验的新发布”；抓取失败则写“覆盖不完整”。全文未取得不使用标题/摘要补造结论。

来源监控只是哈希与可用性检查，不做版本公告判断。新闻/标准/实现/硬件交付/作者结果/自己的测量分别登记，不互相升级。

日报引用当日归档监控报告，而非会被后续日期覆盖的全局 latest 文件。可用 `--output research\monitor-YYYY-MM-DD.json` 生成当日报告，并将 progress.monitor_report 指向它；同日多次核验可增量更新或保留明确命名的快照，禁止重复创建同一天的日报。

## 运行与恢复

自动化每日北京时间 22:30，从真实远端状态恢复；不新增 GitHub 定时任务/后台循环。Local 运行依赖本机/App/网络与认证。中断或认证失败必须报告；下一次从最后成功提交继续，不伪称已补跑。

进度保存活动分支（默认 `"main"`）与 `active_pr`（直推主分支模式下为 `null`），不在文件里填写其自身最终 commit SHA 形成循环提交。远端 SHA、调度运行结果以工具读回核验。
