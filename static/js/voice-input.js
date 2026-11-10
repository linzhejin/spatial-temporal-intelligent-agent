/* ========================================================================
   珞珈智行 · 语音输入（voice-input.js）—— 按住说话
   · APK 内：Android 原生 SpeechRecognizer（显式绑定厂商 RecognitionService）
       WhuWalkerVoice.press() / release() / cancel()
       原生通过 window.__whuWalkerVoiceCallback(json) 回传：
       {event:'start'|'level'|'partial'|'final'|'end'|'error', ...}
   · 浏览器/PWA：Web Speech API（webkitSpeechRecognition，zh-CN，实时转写）
   · 聆听浮层为本文件自绘的紧凑小卡片（不弹任何系统全屏听写面板）
   · Fail-soft：任何异常只走回调，不抛错
   ======================================================================== */
(function () {
    'use strict';

    var SR = window.SpeechRecognition || window.webkitSpeechRecognition || null;

    // ===== 一轮按住的状态 =====
    var active = false;        // 按住周期内（按下 → final/end/error/cancel）
    var cancelled = false;     // 上滑取消
    var released = false;      // 已松手
    var hooks = null;
    var cardEl = null;
    var cardIconEl = null;
    var cardTextEl = null;
    var cardInterimEl = null;
    var cardHintEl = null;

    // 浏览器 SpeechRecognition
    var recognition = null;
    var srStarted = false;     // onstart 已触发
    var pendingRelease = false; // onstart 前松手
    var finalFired = false;

    function isWeChat() {
        return /MicroMessenger/i.test(navigator.userAgent || '');
    }

    function apkBridge() {
        try {
            if (window.WhuWalkerVoice && typeof window.WhuWalkerVoice.press === 'function') {
                return window.WhuWalkerVoice;
            }
        } catch (e) { /* ignore */ }
        return null;
    }

    // ================= 紧凑聆听浮层 =================
    function ensureCard() {
        if (cardEl) return cardEl;
        cardEl = document.createElement('div');
        cardEl.className = 'voice-hold-card';
        cardEl.setAttribute('aria-live', 'polite');
        cardEl.innerHTML =
            '<div class="vhc-icon"><span class="vhc-dot"></span>'
            + '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            + 'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
            + '<path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>'
            + '<path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>'
            + '<line x1="12" y1="19" x2="12" y2="23"></line>'
            + '<line x1="8" y1="23" x2="16" y2="23"></line></svg></div>'
            + '<div class="vhc-text">正在聆听</div>'
            + '<div class="vhc-interim"></div>'
            + '<div class="vhc-hint">松开发送 · 上滑取消</div>';
        document.body.appendChild(cardEl);
        cardIconEl = cardEl.querySelector('.vhc-icon');
        cardTextEl = cardEl.querySelector('.vhc-text');
        cardInterimEl = cardEl.querySelector('.vhc-interim');
        cardHintEl = cardEl.querySelector('.vhc-hint');
        return cardEl;
    }

    function showCard() {
        ensureCard();
        cardEl.classList.remove('show', 'cancel');
        // 强制 reflow 再进 show，保证过渡动画生效
        void cardEl.offsetWidth;
        cardTextEl.textContent = '正在聆听';
        cardInterimEl.textContent = '';
        cardHintEl.textContent = '松开发送 · 上滑取消';
        cardEl.classList.add('show');
    }

    function setCancelUI(on) {
        if (!cardEl) return;
        cardEl.classList.toggle('cancel', !!on);
        cardTextEl.textContent = on ? '松开取消' : '正在聆听';
        cardHintEl.textContent = on ? '松开取消发送' : '松开发送 · 上滑取消';
    }

    function setInterim(text) {
        if (!cardEl || !text) return;
        cardInterimEl.textContent = text;
    }

    function setLevel(v) {
        if (!cardIconEl) return;
        var scale = 1 + Math.max(0, Math.min(1, v)) * 0.35;
        cardIconEl.style.transform = 'scale(' + scale.toFixed(2) + ')';
    }

    function hideCard() {
        if (cardEl) cardEl.classList.remove('show', 'cancel');
        if (cardIconEl) cardIconEl.style.transform = '';
    }

    // ================= 原生回调入口 =================
    window.__whuWalkerVoiceCallback = function (payload) {
        var ev;
        try {
            ev = typeof payload === 'string' ? JSON.parse(payload) : payload;
        } catch (e) {
            console.warn('[ASR] 原生回调解析失败:', payload);
            return;
        }
        if (!ev || !active) return;

        if (ev.event === 'start') {
            showCard();
            if (hooks.onStateChange) hooks.onStateChange('listening');
        } else if (ev.event === 'level') {
            setLevel(ev.value);
        } else if (ev.event === 'partial') {
            if (ev.text) {
                setInterim(ev.text);
                if (hooks.onInterim) hooks.onInterim(ev.text);
            }
        } else if (ev.event === 'final') {
            var text = (ev.text || '').trim();
            active = false;
            hideCard();
            if (hooks.onStateChange) hooks.onStateChange('idle');
            if (text && hooks.onFinal) hooks.onFinal(text);
            hooks = null;
        } else if (ev.event === 'error') {
            active = false;
            hideCard();
            if (hooks.onStateChange) hooks.onStateChange('idle');
            if (hooks.onError) hooks.onError(ev.message || '语音识别失败');
            hooks = null;
        } else if (ev.event === 'end') {
            active = false;
            hideCard();
            if (hooks.onStateChange) hooks.onStateChange('idle');
            hooks = null;
        }
    };

    // ================= 浏览器（Web Speech）按住语义 =================
    function startViaBrowser(cb) {
        if (!SR) {
            if (cb.onUnsupported) cb.onUnsupported(isWeChat() ? 'wechat' : 'unsupported');
            return;
        }
        try {
            recognition = new SR();
        } catch (e) {
            if (cb.onUnsupported) cb.onUnsupported('unsupported');
            return;
        }
        recognition.lang = 'zh-CN';
        recognition.interimResults = true;
        recognition.continuous = false;
        recognition.maxAlternatives = 1;
        srStarted = false;
        pendingRelease = false;
        finalFired = false;

        recognition.onstart = function () {
            srStarted = true;
            showCard();
            if (cb.onStateChange) cb.onStateChange('listening');
            // 点按极快：start 回来时手指已经松开
            if (pendingRelease) {
                try { recognition.stop(); } catch (e) { /* ignore */ }
            }
        };

        recognition.onresult = function (e) {
            var interim = '';
            var finalText = '';
            for (var i = e.resultIndex; i < e.results.length; i++) {
                var r = e.results[i];
                if (r.isFinal) finalText += r[0].transcript;
                else interim += r[0].transcript;
            }
            if (interim) {
                setInterim(interim);
                if (cb.onInterim) cb.onInterim(interim);
            }
            if (finalText && !finalFired) {
                finalFired = true;
                active = false;
                hideCard();
                if (cb.onFinal) cb.onFinal(finalText.trim());
            }
        };

        recognition.onerror = function (e) {
            if (cancelled) return;  // abort() 会跟一个 error，忽略
            // 松手后没说话（no-speech）/极短录音：安静结束，不打扰
            if (released && (e.error === 'no-speech' || e.error === 'aborted')) return;
            if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
                if (cb.onUnsupported) cb.onUnsupported('permission');
                return;
            }
            var map = {
                'audio-capture': '找不到麦克风设备',
                'network': '语音服务网络异常',
            };
            if (cb.onError) cb.onError(map[e.error] || '没有听清，请按住重说');
        };

        recognition.onend = function () {
            srStarted = false;
            if (!active) return;
            active = false;
            hideCard();
            if (cb.onStateChange) cb.onStateChange('idle');
            hooks = null;
        };

        try {
            recognition.start();
        } catch (e) {
            if (cb.onError) cb.onError('语音启动中，请稍后再试');
        }
    }

    var VoiceInput = {
        /**
         * 按下麦克风。
         * @param {Object} cb onInterim/onFinal/onStateChange/onLevel/onUnsupported/onError
         */
        press: function (cb) {
            if (active) return;
            hooks = cb || {};
            active = true;
            cancelled = false;
            released = false;

            var bridge = apkBridge();
            if (bridge) {
                // 先把卡片立起来（权限弹窗时也有反馈），start 事件回来时刷新
                showCard();
                try {
                    bridge.press();
                } catch (e) {
                    active = false;
                    hideCard();
                    if (hooks.onUnsupported) hooks.onUnsupported('unsupported');
                    hooks = null;
                }
            } else {
                showCard();
                startViaBrowser(hooks);
            }
        },

        /** 松手：结束录音、等待识别文本（浏览器立即 stop，原生等 onResults） */
        release: function () {
            if (!active || released) return;
            released = true;
            var bridge = apkBridge();
            if (bridge) {
                try { bridge.release(); } catch (e) { /* ignore */ }
                return;
            }
            if (recognition) {
                if (srStarted) {
                    try { recognition.stop(); } catch (e) { /* ignore */ }
                } else {
                    pendingRelease = true;
                }
            }
        },

        /** 上滑取消：丢弃结果、关闭浮层 */
        cancel: function () {
            if (!active) return;
            cancelled = true;
            released = true;
            var bridge = apkBridge();
            if (bridge) {
                try { bridge.cancel(); } catch (e) { /* ignore */ }
            } else if (recognition) {
                try { recognition.abort(); } catch (e) { /* ignore */ }
            }
            active = false;
            hideCard();
            if (hooks && hooks.onStateChange) hooks.onStateChange('idle');
            hooks = null;
        },

        /** 手指仍按着但取消手势切换（dy 变化时调用），只改 UI */
        setCancelHint: function (willCancel) {
            if (!active) return;
            var bridge = apkBridge();
            setCancelUI(!!willCancel);
            // 原生侧同步取消标志：越过阈值时就告知，避免松手瞬间竞态丢结果
            if (bridge) {
                try {
                    if (willCancel) bridge.cancel();
                    // 注意：cancel 一旦发出不可恢复，由调用方保证越过阈值后不再回退
                } catch (e) { /* ignore */ }
            }
        },

        isHolding: function () { return active; },

        /** {ok:boolean, reason?:'wechat'|'unsupported'} */
        capability: function () {
            if (apkBridge() || SR) return { ok: true };
            return { ok: false, reason: isWeChat() ? 'wechat' : 'unsupported' };
        },
    };

    window.WhuWalkerVoiceInput = VoiceInput;
})();
