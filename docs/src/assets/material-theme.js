/* Material Design Theme JavaScript Enhancements */
/* Adds smooth scrolling, enhanced interactions, and Material Design behaviors */

document.addEventListener('DOMContentLoaded', function() {

    // Smooth scrolling for anchor links
    function initSmoothScrolling() {
        const links = document.querySelectorAll('a[href^="#"]');
        links.forEach(link => {
            link.addEventListener('click', function(e) {
                e.preventDefault();
                const targetId = this.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    targetElement.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    // Add ripple effects to clickable elements (Material Design)
    function addRippleEffects() {
        const clickableElements = document.querySelectorAll('.tocitem, button, .btn');

        clickableElements.forEach(element => {
            element.addEventListener('click', function(e) {
                const ripple = document.createElement('span');
                const rect = this.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;

                ripple.style.width = ripple.style.height = size + 'px';
                ripple.style.left = x + 'px';
                ripple.style.top = y + 'px';
                ripple.classList.add('ripple');

                this.appendChild(ripple);

                setTimeout(() => {
                    ripple.remove();
                }, 600);
            });
        });
    }

    // Enhanced sidebar behavior
    function enhanceSidebar() {
        const sidebar = document.querySelector('.docs-sidebar');
        const mainContent = document.querySelector('.docs-main');

        if (!sidebar || !mainContent) return;

        // Add mobile menu toggle functionality
        let menuToggle = document.querySelector('.mobile-menu-toggle');
        if (!menuToggle) {
            menuToggle = document.createElement('button');
            menuToggle.classList.add('mobile-menu-toggle');
            menuToggle.innerHTML = '☰';
            menuToggle.style.cssText = `
                display: none;
                position: fixed;
                top: 16px;
                left: 16px;
                z-index: 1001;
                background: var(--md-primary-color);
                color: white;
                border: none;
                padding: 12px;
                border-radius: 8px;
                font-size: 18px;
                cursor: pointer;
                box-shadow: var(--md-elevation-2);
            `;
            document.body.appendChild(menuToggle);
        }

        // Show/hide menu toggle on mobile
        function updateMobileMenu() {
            if (window.innerWidth <= 768) {
                menuToggle.style.display = 'block';
            } else {
                menuToggle.style.display = 'none';
                sidebar.classList.remove('is-open');
            }
        }

        // Toggle sidebar on mobile
        menuToggle.addEventListener('click', function() {
            sidebar.classList.toggle('is-open');
        });

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', function(e) {
            if (window.innerWidth <= 768 &&
                sidebar.classList.contains('is-open') &&
                !sidebar.contains(e.target) &&
                !menuToggle.contains(e.target)) {
                sidebar.classList.remove('is-open');
            }
        });

        // Initial setup and resize handler
        updateMobileMenu();
        window.addEventListener('resize', updateMobileMenu);
    }

    // Add copy-to-clipboard functionality for code blocks
    function addCodeCopyButtons() {
        const codeBlocks = document.querySelectorAll('pre code');

        codeBlocks.forEach(block => {
            const pre = block.parentElement;
            if (pre.tagName !== 'PRE') return;

            const button = document.createElement('button');
            button.classList.add('copy-code-btn');
            button.innerHTML = '📋';
            button.title = 'Copy to clipboard';
            button.style.cssText = `
                position: absolute;
                top: 8px;
                right: 8px;
                background: var(--md-surface);
                border: 1px solid var(--md-outline-variant);
                border-radius: 4px;
                padding: 4px 8px;
                cursor: pointer;
                font-size: 14px;
                opacity: 0;
                transition: opacity 0.2s ease;
            `;

            pre.style.position = 'relative';
            pre.appendChild(button);

            // Show button on hover
            pre.addEventListener('mouseenter', () => button.style.opacity = '1');
            pre.addEventListener('mouseleave', () => button.style.opacity = '0');

            // Copy functionality
            button.addEventListener('click', async function() {
                try {
                    await navigator.clipboard.writeText(block.textContent);
                    this.innerHTML = '✅';
                    setTimeout(() => this.innerHTML = '📋', 2000);
                } catch (err) {
                    console.error('Failed to copy: ', err);
                    this.innerHTML = '❌';
                    setTimeout(() => this.innerHTML = '📋', 2000);
                }
            });
        });
    }

    // Add scroll-to-top functionality
    function addScrollToTop() {
        const scrollBtn = document.createElement('button');
        scrollBtn.classList.add('scroll-to-top');
        scrollBtn.innerHTML = '↑';
        scrollBtn.title = 'Scroll to top';
        scrollBtn.style.cssText = `
            position: fixed;
            bottom: 24px;
            right: 24px;
            background: var(--md-primary-color);
            color: white;
            border: none;
            border-radius: 50%;
            width: 56px;
            height: 56px;
            font-size: 24px;
            cursor: pointer;
            box-shadow: var(--md-elevation-3);
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            z-index: 1000;
        `;

        document.body.appendChild(scrollBtn);

        // Show/hide based on scroll position
        window.addEventListener('scroll', function() {
            if (window.pageYOffset > 300) {
                scrollBtn.style.opacity = '1';
                scrollBtn.style.visibility = 'visible';
            } else {
                scrollBtn.style.opacity = '0';
                scrollBtn.style.visibility = 'hidden';
            }
        });

        // Scroll to top functionality
        scrollBtn.addEventListener('click', function() {
            window.scrollTo({
                top: 0,
                behavior: 'smooth'
            });
        });
    }

    // Add loading animation for page transitions
    function addLoadingAnimations() {
        // Fade in content after load
        const mainContent = document.querySelector('.docs-main main');
        if (mainContent) {
            mainContent.style.opacity = '0';
            mainContent.style.transform = 'translateY(20px)';
            mainContent.style.transition = 'opacity 0.5s ease, transform 0.5s ease';

            setTimeout(() => {
                mainContent.style.opacity = '1';
                mainContent.style.transform = 'translateY(0)';
            }, 100);
        }
    }

    // Enhance search functionality (if search exists)
    function enhanceSearch() {
        const searchInput = document.querySelector('.docs-search input');
        if (!searchInput) return;

        // Add search icon
        const searchContainer = searchInput.parentElement;
        const searchIcon = document.createElement('span');
        searchIcon.innerHTML = '🔍';
        searchIcon.style.cssText = `
            position: absolute;
            right: 16px;
            top: 50%;
            transform: translateY(-50%);
            color: var(--md-on-surface-variant);
            pointer-events: none;
        `;

        searchContainer.style.position = 'relative';
        searchContainer.appendChild(searchIcon);

        searchInput.style.paddingRight = '48px';
    }

    // Add Material Design focus indicators
    function addFocusIndicators() {
        const style = document.createElement('style');
        style.textContent = `
            .ripple {
                position: absolute;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.6);
                animation: ripple-animation 0.6s linear;
                pointer-events: none;
            }

            @keyframes ripple-animation {
                to {
                    transform: scale(4);
                    opacity: 0;
                }
            }

            *:focus {
                outline: 2px solid var(--md-primary-color);
                outline-offset: 2px;
            }

            input:focus, textarea:focus {
                outline: none;
                box-shadow: 0 0 0 2px var(--md-primary-color);
            }
        `;
        document.head.appendChild(style);
    }

    // Initialize all enhancements
    function init() {
        initSmoothScrolling();
        addRippleEffects();
        enhanceSidebar();
        addCodeCopyButtons();
        addScrollToTop();
        addLoadingAnimations();
        enhanceSearch();
        addFocusIndicators();
        initSyntaxHighlighting();

        console.log('Material Design theme enhancements loaded');
    }

    // Initialize syntax highlighting
    function initSyntaxHighlighting() {
        // Load and initialize Highlight.js
        if (typeof require !== 'undefined') {
            require(['https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js'], function(hljs) {
                // Configure Highlight.js
                hljs.configure({
                    languages: ['julia', 'javascript', 'python', 'bash', 'css', 'html', 'json', 'markdown']
                });
                
                // Highlight all code blocks
                hljs.highlightAll();
                
                console.log('Syntax highlighting initialized');
            });
        } else {
            // Fallback: load via script tag if require.js is not available
            const script = document.createElement('script');
            script.src = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js';
            script.onload = function() {
                if (window.hljs) {
                    window.hljs.configure({
                        languages: ['julia', 'javascript', 'python', 'bash', 'css', 'html', 'json', 'markdown']
                    });
                    window.hljs.highlightAll();
                    console.log('Syntax highlighting initialized (fallback)');
                }
            };
            document.head.appendChild(script);
        }
    }

    // Run initialization
    init();
});

// Add some utility functions for dynamic content
window.MaterialTheme = {
    // Function to reinitialize enhancements after dynamic content loads
    refresh: function() {
        // Re-run specific initializations that might be needed
        const event = new Event('DOMContentLoaded');
        document.dispatchEvent(event);
    },

    // Function to toggle dark mode (if needed later)
    toggleDarkMode: function() {
        document.documentElement.classList.toggle('dark-mode');
        localStorage.setItem('dark-mode', document.documentElement.classList.contains('dark-mode'));
    }
};
