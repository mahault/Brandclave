
        // Store data globally for modal access
        var allTrends = [];
        var allMoves = [];

        function openTrendModal(i) {
            var t = allTrends[i];
            if (!t) return;
            document.getElementById('modal-header').className = 'modal-header';
            document.getElementById('modal-title').textContent = t.name || t.trend_name || 'Unnamed Trend';
            var score = t.strength_score ? Math.round(t.strength_score * 100) + '%' : 'N/A';
            var whiteSpace = t.white_space_score ? Math.round(t.white_space_score * 100) : 0;
            document.getElementById('modal-meta').innerHTML = 'Strength: ' + score + ' | White Space: ' + whiteSpace + '% | ' + (t.volume || 0) + ' sources';
            var h = '';
            if (t.description) h += '<div class="modal-section"><h3>Description</h3><p>' + t.description + '</p></div>';
            if (t.why_it_matters) h += '<div class="modal-section"><h3>Why It Matters</h3><p>' + t.why_it_matters + '</p></div>';
            // White Space Analysis section
            if (t.white_space_score !== undefined) {
                var ws = Math.round((t.white_space_score || 0) * 100);
                var wsClass = ws >= 70 ? 'white-space-high' : ws >= 40 ? 'white-space-medium' : 'white-space-low';
                var wsLabel = ws >= 70 ? 'High Opportunity - underserved market' : ws >= 40 ? 'Moderate Opportunity' : 'Low - competitive market';
                h += '<div class="modal-section"><h3>🎯 White Space Analysis</h3>';
                h += '<p><span class="white-space-badge ' + wsClass + '" style="font-size:1em;padding:6px 12px;">' + ws + '% - ' + wsLabel + '</span></p>';
                if (t.region) h += '<p style="margin-top:10px;"><strong>Region:</strong> ' + t.region + '</p>';
                if (t.audience_segment) h += '<p><strong>Segment:</strong> ' + t.audience_segment + '</p>';
                h += '</div>';
            }
            if (t.topics && t.topics.length) {
                var topicsHtml = '';
                for (var ti = 0; ti < t.topics.length; ti++) { topicsHtml += '<span class="topic-tag">' + t.topics[ti] + '</span>'; }
                h += '<div class="modal-section"><h3>Topics</h3><div>' + topicsHtml + '</div></div>';
            }
            if (t.sample_quotes && t.sample_quotes.length) {
                h += '<div class="modal-section"><h3>Source Quotes</h3>';
                for (var qi = 0; qi < t.sample_quotes.length; qi++) { h += '<div class="source-quote">"' + t.sample_quotes[qi] + '"</div>'; }
                h += '</div>';
            }
            document.getElementById('modal-body').innerHTML = h || '<p>No additional details.</p>';
            document.getElementById('modal-overlay').classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function openMoveModal(i) {
            var m = allMoves[i];
            if (!m) return;
            document.getElementById('modal-header').className = 'modal-header move-header';
            document.getElementById('modal-title').textContent = m.title || 'Untitled';

            // Enhanced meta with market and investment
            var metaParts = [m.company || 'Unknown', m.move_type ? m.move_type.replace('_', ' ') : 'Move'];
            if (m.market) metaParts.push('📍 ' + m.market);
            if (m.investment_amount) metaParts.push('💰 ' + m.investment_amount);
            document.getElementById('modal-meta').innerHTML = metaParts.join(' | ');

            var h = '';

            // Summary
            if (m.summary) h += '<div class="modal-section"><h3>Summary</h3><p>' + m.summary + '</p></div>';

            // Why It Matters
            if (m.why_it_matters) h += '<div class="modal-section"><h3>Why It Matters</h3><p>' + m.why_it_matters + '</p></div>';

            // Strategic Implications
            if (m.strategic_implications && m.strategic_implications.length > 0) {
                h += '<div class="modal-section"><h3>Strategic Implications</h3><ul style="margin:0;padding-left:20px;">';
                for (var i = 0; i < m.strategic_implications.length; i++) {
                    h += '<li style="margin-bottom:6px;">' + m.strategic_implications[i] + '</li>';
                }
                h += '</ul></div>';
            }

            // Competitive Impact
            if (m.competitive_impact) {
                h += '<div class="modal-section"><h3>Competitive Impact</h3><p>' + m.competitive_impact + '</p></div>';
            }

            // Source
            if (m.source_url) h += '<div class="modal-section"><h3>Source</h3><p><a href="' + m.source_url + '" target="_blank" style="color:#667eea;">' + (m.source_name || 'View article') + '</a></p></div>';

            document.getElementById('modal-body').innerHTML = h || '<p>No additional details.</p>';
            document.getElementById('modal-overlay').classList.add('active');
            document.body.style.overflow = 'hidden';
        }

        function closeModal(e) {
            if (e && e.target !== e.currentTarget) return;
            document.getElementById('modal-overlay').classList.remove('active');
            document.body.style.overflow = '';
        }
        document.addEventListener('keydown', function(e) { if (e.key === 'Escape') closeModal(); });


        function showTab(tabId) {
            var sections = document.querySelectorAll('.section');
            for (var i = 0; i < sections.length; i++) { sections[i].classList.remove('active'); }
            var tabs = document.querySelectorAll('.tab');
            for (var j = 0; j < tabs.length; j++) { tabs[j].classList.remove('active'); }
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }

        function setStatus(icon, text) {
            document.getElementById('status-icon').textContent = icon;
            document.getElementById('status-text').textContent = text;
        }

        function truncate(str, len) {
            if (!str) return '';
            return str.length > len ? str.substring(0, len) + '...' : str;
        }

        async function loadAllData() {
            setStatus('⏳', 'Loading data...');

            try {
                // Fetch all data in parallel
                var responses = await Promise.all([
                    fetch('/api/monitoring/metrics'),
                    fetch('/api/social-pulse?limit=10'),
                    fetch('/api/hotelier-bets?limit=10'),
                    fetch('/api/monitoring/content?limit=30'),
                    fetch('/api/monitoring/scrapers')
                ]);
                var metricsRes = responses[0];
                var trendsRes = responses[1];
                var movesRes = responses[2];
                var contentRes = responses[3];
                var scrapersRes = responses[4];

                var metrics = await metricsRes.json();
                var trendsData = await trendsRes.json();
                var movesData = await movesRes.json();
                var contentData = await contentRes.json();
                var scrapers = await scrapersRes.json();

                // Update metrics
                document.getElementById('m-content').textContent = (metrics.total_content ? metrics.total_content.toLocaleString() : '0');
                document.getElementById('m-processed').textContent = (metrics.processed_content ? metrics.processed_content.toLocaleString() : '0');
                document.getElementById('m-trends').textContent = metrics.trends_count || '0';
                document.getElementById('m-moves').textContent = metrics.moves_count || '0';

                // Render trends
                allTrends = trendsData.trends || [];
                var trends = allTrends;
                if (trends.length > 0) {
                    document.getElementById('latest-trend').innerHTML = renderTrend(trends[0], 0);
                    var trendsHtml = '';
                    for (var ti = 0; ti < trends.length; ti++) { trendsHtml += renderTrend(trends[ti], ti); }
                    document.getElementById('trends-list').innerHTML = trendsHtml;
                } else {
                    document.getElementById('latest-trend').innerHTML = '<div class="empty"><div class="icon">📈</div>No trends yet. Run POPULATE_DATA.bat</div>';
                    document.getElementById('trends-list').innerHTML = '<div class="empty"><div class="icon">📈</div>No trends yet</div>';
                }

                // Render moves
                allMoves = movesData.moves || [];
                var moves = allMoves;
                if (moves.length > 0) {
                    document.getElementById('latest-move').innerHTML = renderMove(moves[0], 0);
                    var movesHtml = '';
                    for (var mi = 0; mi < moves.length; mi++) { movesHtml += renderMove(moves[mi], mi); }
                    document.getElementById('moves-list').innerHTML = movesHtml;
                } else {
                    document.getElementById('latest-move').innerHTML = '<div class="empty"><div class="icon">♟️</div>No moves yet. Run POPULATE_DATA.bat</div>';
                    document.getElementById('moves-list').innerHTML = '<div class="empty"><div class="icon">♟️</div>No moves yet</div>';
                }

                // Render content
                var content = contentData.items || [];
                if (content.length > 0) {
                    var contentHtml = '';
                    for (var ci = 0; ci < content.length; ci++) { contentHtml += renderContent(content[ci]); }
                    document.getElementById('content-list').innerHTML = contentHtml;
                } else {
                    document.getElementById('content-list').innerHTML = '<div class="empty"><div class="icon">📰</div>No content yet</div>';
                }

                // Render scrapers
                if (scrapers.length > 0) {
                    var scrapersHtml = '<table><thead><tr><th>Source</th><th>Total</th><th>Last Run</th><th>New Items</th><th>Status</th></tr></thead><tbody>';
                    for (var si = 0; si < scrapers.length; si++) {
                        var s = scrapers[si];
                        var lastRunTime = s.last_run_at ? new Date(s.last_run_at).toLocaleString() : 'Never';
                        var statusClass = s.last_run_status === 'completed' ? 'success' : 'warning';
                        var statusText = s.last_run_status || 'N/A';
                        scrapersHtml += '<tr><td><strong>' + s.source + '</strong></td><td>' + s.total_items.toLocaleString() + '</td><td>' + lastRunTime + '</td><td>' + (s.last_run_items || 0) + '</td><td><span class="badge badge-' + statusClass + '">' + statusText + '</span></td></tr>';
                    }
                    scrapersHtml += '</tbody></table>';
                    document.getElementById('scrapers-list').innerHTML = scrapersHtml;
                } else {
                    document.getElementById('scrapers-list').innerHTML = '<div class="empty"><div class="icon">🔧</div>No scraper data</div>';
                }

                // Load scanned properties (don't wait for it)
                loadScannedProperties();

                setStatus('✅', 'Data loaded at ' + new Date().toLocaleTimeString());

            } catch (err) {
                console.error('Load error:', err);
                setStatus('❌', 'Error: ' + err.message);
                document.getElementById('latest-trend').innerHTML = '<div class="error">Failed to load data: ' + err.message + '</div>';
                document.getElementById('latest-move').innerHTML = '<div class="error">Failed to load data: ' + err.message + '</div>';
            }
        }

        function renderTrend(t, idx) {
            var name = t.name || t.trend_name || 'Unnamed Trend';
            var score = t.strength_score ? Math.round(t.strength_score * 100) + '%' : 'N/A';
            var whiteSpace = t.white_space_score ? Math.round(t.white_space_score * 100) : 0;
            var wsClass = whiteSpace >= 70 ? 'white-space-high' : whiteSpace >= 40 ? 'white-space-medium' : 'white-space-low';
            var wsLabel = whiteSpace >= 70 ? 'High Opportunity' : whiteSpace >= 40 ? 'Moderate' : 'Low';
            var isSaved = isProjectSaved(t.id);
            var savedClass = isSaved ? 'saved' : '';
            var savedText = isSaved ? '✓ Saved' : '💾 Save';

            return '<div class="trend-card" data-trend-id="' + t.id + '">' +
                '<div onclick="openTrendModal(' + idx + ')" style="cursor:pointer;">' +
                '<h3>' + truncate(name, 60) + '</h3>' +
                '<p>' + truncate(t.description || t.why_it_matters || '', 200) + '</p>' +
                '<div class="trend-meta">' +
                '<span class="white-space-badge ' + wsClass + '">🎯 ' + whiteSpace + '% ' + wsLabel + '</span>' +
                ' | Strength: ' + score + ' | ' + (t.volume || 0) + ' sources' +
                '</div></div>' +
                '<div class="trend-actions">' +
                '<button class="trend-action-btn btn-save ' + savedClass + '" onclick="event.stopPropagation(); toggleSaveProject(' + idx + ')">' + savedText + '</button>' +
                '<button class="trend-action-btn btn-brand" onclick="event.stopPropagation(); turnIntoBrand(' + idx + ')">🚀 Build a Brand</button>' +
                '</div></div>';
        }

        function renderMove(m, idx) {
            var isSaved = isMoveSaved(m.id);
            var moveTypeBadge = m.move_type ? '<span class="move-type-badge">' + m.move_type.replace('_', ' ') + '</span>' : '';
            var marketBadge = m.market ? '<span class="market-badge">📍 ' + m.market + '</span>' : '';
            var savedClass = isSaved ? 'saved' : '';
            var savedText = isSaved ? '✓ Saved' : '💾 Save';

            return '<div class="move-card" onclick="openMoveModal(' + idx + ')">' +
                '<div class="move-badges">' + moveTypeBadge + marketBadge + '</div>' +
                '<h3>' + truncate(m.title || 'Untitled', 60) + '</h3>' +
                '<div class="company">' + (m.company || 'Unknown') + '</div>' +
                '<p>' + truncate(m.summary || m.why_it_matters || '', 180) + '</p>' +
                '<div class="move-actions">' +
                '<button class="move-action-btn btn-save ' + savedClass + '" onclick="event.stopPropagation(); toggleSaveMove(' + idx + ')">' + savedText + '</button>' +
                '</div></div>';
        }

        function renderContent(c) {
            var dateStr = c.published_at ? ' • ' + new Date(c.published_at).toLocaleDateString() : '';
            return '<div class="content-item">' +
                '<h4>' + truncate(c.title || 'Untitled', 70) + '</h4>' +
                '<p>' + truncate(c.content || '', 150) + '</p>' +
                '<div class="meta"><span class="source">' + (c.source || 'unknown') + '</span>' + dateStr + '</div>' +
                '</div>';
        }

        // =============================================
        // Demand Scan Functions
        // =============================================

        var allProperties = [];

        async function scanProperty() {
            var urlInput = document.getElementById('property-url-input');
            var url = urlInput.value.trim();

            if (!url) {
                showScanStatus('error', 'Please enter a valid URL');
                return;
            }

            // Validate URL format
            try {
                new URL(url);
            } catch (e) {
                showScanStatus('error', 'Invalid URL format. Please enter a complete URL including https://');
                return;
            }

            var scanBtn = document.getElementById('scan-btn');
            scanBtn.disabled = true;
            scanBtn.textContent = 'Scanning...';
            showScanStatus('info', 'Analyzing property website... This may take 30-60 seconds.');

            try {
                var response = await fetch('/api/demand-scan', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url })
                });

                var data = await response.json();

                if (response.ok) {
                    if (data.status === 'exists') {
                        showScanStatus('warning', 'Property was previously scanned. Showing existing analysis.');
                    } else {
                        showScanStatus('success', 'Property analyzed successfully!');
                    }
                    urlInput.value = '';
                    loadScannedProperties();
                } else {
                    showScanStatus('error', data.detail || 'Failed to scan property. Please check the URL and try again.');
                }
            } catch (err) {
                console.error('Scan error:', err);
                showScanStatus('error', 'Network error. Please try again.');
            } finally {
                scanBtn.disabled = false;
                scanBtn.textContent = 'Scan Property';
            }
        }

        function showScanStatus(type, message) {
            var statusEl = document.getElementById('scan-status');
            var colors = {
                error: { bg: '#fee2e2', color: '#991b1b' },
                success: { bg: '#d4edda', color: '#155724' },
                warning: { bg: '#fff3cd', color: '#856404' },
                info: { bg: '#e0e7ff', color: '#4338ca' }
            };
            var style = colors[type] || colors.info;
            statusEl.style.display = 'block';
            statusEl.style.background = style.bg;
            statusEl.style.color = style.color;
            statusEl.textContent = message;

            if (type !== 'info') {
                setTimeout(function() { statusEl.style.display = 'none'; }, 5000);
            }
        }

        async function loadScannedProperties() {
            try {
                var response = await fetch('/api/demand-scan?limit=20');
                var data = await response.json();
                allProperties = data.properties || [];

                var container = document.getElementById('properties-list');
                if (allProperties.length > 0) {
                    var propsHtml = '';
                    for (var pi = 0; pi < allProperties.length; pi++) { propsHtml += renderPropertyCard(allProperties[pi], pi); }
                    container.innerHTML = propsHtml;
                } else {
                    container.innerHTML = '<div class="empty"><div class="icon">🏨</div>No properties scanned yet. Enter a URL above to analyze a property.</div>';
                }
            } catch (err) {
                console.error('Load properties error:', err);
            }
        }

        function renderPropertyCard(p, idx) {
            // Calculate demand score as 0-100
            var score = p.demand_fit_score ? Math.round(p.demand_fit_score * 100) : 0;
            var scoreClass = score >= 70 ? 'demand-high' : score >= 40 ? 'demand-medium' : 'demand-low';
            var scoreLabel = score >= 70 ? 'High Fit' : score >= 40 ? 'Moderate Fit' : 'Low Fit';

            // Experience gaps (top 3)
            var gaps = (p.experience_gaps || []).slice(0, 3);
            var gapsHtml = '';
            if (gaps.length > 0) {
                for (var gi = 0; gi < gaps.length; gi++) {
                    gapsHtml += '<span class="gap-item">' + truncate(gaps[gi].split(' (')[0], 50) + '</span>';
                }
            } else {
                gapsHtml = '<span style="color:#888;font-size:0.85em;">No major gaps identified</span>';
            }

            // Opportunity lanes (top 2)
            var opportunities = (p.opportunity_lanes || []).slice(0, 2);
            var oppsHtml = '';
            if (opportunities.length > 0) {
                for (var oi = 0; oi < opportunities.length; oi++) {
                    oppsHtml += '<div class="opportunity-item">' + truncate(opportunities[oi], 120) + '</div>';
                }
            } else {
                oppsHtml = '<span style="color:#888;font-size:0.85em;">No opportunities identified</span>';
            }

            // Misalignment flags
            var flags = p.positioning_misalignment_flags || [];
            var flagsHtml = '';
            if (flags.length > 0) {
                flagsHtml = '<div class="property-section"><div class="property-section-title">Positioning Issues</div>';
                for (var fi = 0; fi < flags.length; fi++) {
                    var flagText = flags[fi].split(':')[1] || flags[fi];
                    flagsHtml += '<span class="misalignment-flag">' + truncate(flagText, 100) + '</span>';
                }
                flagsHtml += '</div>';
            }

            // Property themes
            var themes = (p.themes || []).slice(0, 4);
            var themesHtml = '';
            if (themes.length > 0) {
                for (var thi = 0; thi < themes.length; thi++) {
                    themesHtml += '<span class="topic-tag">' + themes[thi] + '</span>';
                }
            }

            return '<div class="property-card" data-property-id="' + p.id + '">' +
                '<div class="property-card-header"><div>' +
                '<h3>' + (p.name || 'Unnamed Property') + '</h3>' +
                '<div class="location">' + (p.location || p.region || 'Location unknown') + '</div>' +
                '<div style="margin-top:8px;">' + themesHtml + '</div></div>' +
                '<div class="demand-score ' + scoreClass + '">' + score + '% ' + scoreLabel + '</div></div>' +
                flagsHtml +
                '<div class="property-section"><div class="property-section-title">Experience Gaps</div>' + gapsHtml + '</div>' +
                '<div class="property-section"><div class="property-section-title">Opportunity Lanes</div>' + oppsHtml + '</div>' +
                '<div class="property-actions">' +
                '<button class="property-action-btn btn-brand" onclick="sendPropertyToBuildBrand(' + idx + ')">🚀 Build a Brand</button>' +
                '<button class="property-action-btn btn-save" onclick="savePropertyToProject(' + idx + ')">💾 Save to Project</button>' +
                '<a href="' + p.url + '" target="_blank" class="property-action-btn btn-save" style="text-decoration:none;">🔗 Visit Site</a>' +
                '</div></div>';
        }

        function sendPropertyToBuildBrand(idx) {
            var p = allProperties[idx];
            if (!p) return;

            // Store property data for Build a Brand page
            var brandData = {
                type: 'property',
                property_name: p.name,
                location: p.location || '',
                segment: p.price_segment || (p.themes && p.themes[0]) || '',
                context: 'Property analysis of ' + p.name + '. Demand fit: ' + Math.round((p.demand_fit_score || 0) * 100) + '%',
                gaps: p.experience_gaps || [],
                opportunities: p.opportunity_lanes || [],
                themes: p.themes || []
            };
            localStorage.setItem('brandclave_brand_prefill', JSON.stringify(brandData));

            // Navigate to Build a Brand page
            window.location.href = '/monitoring/build-a-brand';
        }

        function savePropertyToProject(idx) {
            var p = allProperties[idx];
            if (!p) return;

            // Get existing saved properties
            var saved = JSON.parse(localStorage.getItem('brandclave_saved_properties') || '[]');

            // Check if already saved
            var existingIdx = -1;
            for (var i = 0; i < saved.length; i++) {
                if (saved[i].id === p.id) { existingIdx = i; break; }
            }
            if (existingIdx >= 0) {
                // Remove if already saved
                saved.splice(existingIdx, 1);
                localStorage.setItem('brandclave_saved_properties', JSON.stringify(saved));
                loadScannedProperties();
                return;
            }

            // Add to saved
            saved.push({
                id: p.id,
                type: 'property',
                name: p.name,
                location: p.location,
                demand_fit_score: p.demand_fit_score,
                saved_at: new Date().toISOString()
            });
            localStorage.setItem('brandclave_saved_properties', JSON.stringify(saved));
            loadScannedProperties();
        }

        // Allow Enter key to trigger scan
        document.addEventListener('DOMContentLoaded', function() {
            var urlInput = document.getElementById('property-url-input');
            if (urlInput) {
                urlInput.addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') scanProperty();
                });
            }
        });

        // =============================================
        // Filter Functions
        // =============================================
        var currentFilters = { region: '', audience: '', time: '' };

        async function loadFilterOptions() {
            try {
                var filterResponses = await Promise.all([
                    fetch('/api/social-pulse/regions'),
                    fetch('/api/social-pulse/audiences')
                ]);
                var regionsRes = filterResponses[0];
                var audiencesRes = filterResponses[1];

                var regionsData = await regionsRes.json();
                var audiencesData = await audiencesRes.json();

                // Populate region dropdown
                var regionSelect = document.getElementById('filter-region');
                regionSelect.innerHTML = '<option value="">All Regions</option>';
                var regArr = regionsData.regions || [];
                for (var ri = 0; ri < regArr.length; ri++) {
                    var r = regArr[ri];
                    if (r.region) {
                        regionSelect.innerHTML += '<option value="' + r.region + '">' + r.region + ' (' + r.count + ')</option>';
                    }
                }

                // Populate audience dropdown
                var audienceSelect = document.getElementById('filter-audience');
                audienceSelect.innerHTML = '<option value="">All Segments</option>';
                var audArr = audiencesData.audiences || [];
                for (var ai = 0; ai < audArr.length; ai++) {
                    var a = audArr[ai];
                    if (a.segment) {
                        audienceSelect.innerHTML += '<option value="' + a.segment + '">' + a.segment + ' (' + a.count + ')</option>';
                    }
                }
            } catch (err) {
                console.error('Failed to load filter options:', err);
            }
        }

        async function applyFilters() {
            currentFilters.region = document.getElementById('filter-region').value;
            currentFilters.audience = document.getElementById('filter-audience').value;
            currentFilters.time = document.getElementById('filter-time').value;
            await loadTrendsWithFilters();
        }

        async function loadTrendsWithFilters() {
            var params = new URLSearchParams({ limit: '20' });
            if (currentFilters.region) params.append('region', currentFilters.region);
            if (currentFilters.audience) params.append('audience', currentFilters.audience);

            document.getElementById('trends-list').innerHTML = '<div class="empty"><div class="icon">⏳</div>Loading...</div>';

            try {
                var res = await fetch('/api/social-pulse?' + params.toString());
                var data = await res.json();

                var trends = data.trends || [];

                // Client-side time filtering
                if (currentFilters.time) {
                    var daysAgo = parseInt(currentFilters.time);
                    var cutoff = new Date();
                    cutoff.setDate(cutoff.getDate() - daysAgo);
                    var filteredTrends = [];
                    for (var fi = 0; fi < trends.length; fi++) {
                        if (!trends[fi].first_seen || new Date(trends[fi].first_seen) >= cutoff) {
                            filteredTrends.push(trends[fi]);
                        }
                    }
                    trends = filteredTrends;
                }

                allTrends = trends;

                if (trends.length > 0) {
                    var trendsHtml = '';
                    for (var ti = 0; ti < trends.length; ti++) { trendsHtml += renderTrend(trends[ti], ti); }
                    document.getElementById('trends-list').innerHTML = trendsHtml;
                } else {
                    document.getElementById('trends-list').innerHTML = '<div class="empty"><div class="icon">📈</div>No trends match your filters</div>';
                }
            } catch (err) {
                document.getElementById('trends-list').innerHTML = '<div class="error">Failed to load trends: ' + err.message + '</div>';
            }
        }

        function resetFilters() {
            document.getElementById('filter-region').value = '';
            document.getElementById('filter-audience').value = '';
            document.getElementById('filter-time').value = '';
            currentFilters = { region: '', audience: '', time: '' };
            loadTrendsWithFilters();
        }

        // =============================================
        // Move Filter Functions
        // =============================================
        var moveFilters = { company: '', move_type: '', market: '' };

        async function loadMoveFilterOptions() {
            try {
                var moveFilterResponses = await Promise.all([
                    fetch('/api/hotelier-bets/companies'),
                    fetch('/api/hotelier-bets/move-types'),
                    fetch('/api/hotelier-bets/markets')
                ]);
                var companiesRes = moveFilterResponses[0];
                var moveTypesRes = moveFilterResponses[1];
                var marketsRes = moveFilterResponses[2];

                var companiesData = await companiesRes.json();
                var moveTypesData = await moveTypesRes.json();
                var marketsData = await marketsRes.json();

                // Populate company dropdown
                var companySelect = document.getElementById('filter-company');
                companySelect.innerHTML = '<option value="">All Companies</option>';
                var compArr = companiesData.companies || [];
                for (var ci = 0; ci < compArr.length; ci++) {
                    companySelect.innerHTML += '<option value="' + compArr[ci] + '">' + compArr[ci] + '</option>';
                }

                // Populate move type dropdown
                var moveTypeSelect = document.getElementById('filter-move-type');
                moveTypeSelect.innerHTML = '<option value="">All Move Types</option>';
                var mtArr = moveTypesData.move_types || [];
                for (var mti = 0; mti < mtArr.length; mti++) {
                    var mt = mtArr[mti];
                    // Handle both object format {type, label} and string format
                    var mtValue = typeof mt === 'object' ? mt.type : mt;
                    var mtDisplay = typeof mt === 'object' ? mt.label : mt.replace('_', ' ').replace(/\w/g, function(l) { return l.toUpperCase(); });
                    if (mtValue) moveTypeSelect.innerHTML += '<option value="' + mtValue + '">' + mtDisplay + '</option>';
                }

                // Populate market dropdown
                var marketSelect = document.getElementById('filter-market');
                marketSelect.innerHTML = '<option value="">All Markets</option>';
                var marketArr = marketsData.markets || [];
                for (var mi = 0; mi < marketArr.length; mi++) {
                    if (marketArr[mi]) marketSelect.innerHTML += '<option value="' + marketArr[mi] + '">' + marketArr[mi] + '</option>';
                }
            } catch (err) {
                console.error('Failed to load move filter options:', err);
            }
        }

        async function applyMoveFilters() {
            moveFilters.company = document.getElementById('filter-company').value;
            moveFilters.move_type = document.getElementById('filter-move-type').value;
            moveFilters.market = document.getElementById('filter-market').value;
            await loadMovesWithFilters();
        }

        async function loadMovesWithFilters() {
            var params = new URLSearchParams({ limit: '20' });
            if (moveFilters.company) params.append('company', moveFilters.company);
            if (moveFilters.move_type) params.append('move_type', moveFilters.move_type);
            if (moveFilters.market) params.append('market', moveFilters.market);

            document.getElementById('moves-list').innerHTML = '<div class="empty"><div class="icon">⏳</div>Loading...</div>';

            try {
                var res = await fetch('/api/hotelier-bets?' + params.toString());
                var data = await res.json();

                allMoves = data.moves || [];

                if (allMoves.length > 0) {
                    var movesHtml = '';
                    for (var mi = 0; mi < allMoves.length; mi++) { movesHtml += renderMove(allMoves[mi], mi); }
                    document.getElementById('moves-list').innerHTML = movesHtml;
                } else {
                    document.getElementById('moves-list').innerHTML = '<div class="empty"><div class="icon">♟️</div>No moves match your filters</div>';
                }
                updateMovesSavedCount();
            } catch (err) {
                document.getElementById('moves-list').innerHTML = '<div class="error">Failed to load moves: ' + err.message + '</div>';
            }
        }

        function resetMoveFilters() {
            document.getElementById('filter-company').value = '';
            document.getElementById('filter-move-type').value = '';
            document.getElementById('filter-market').value = '';
            moveFilters = { company: '', move_type: '', market: '' };
            loadMovesWithFilters();
        }

        // =============================================
        // LocalStorage Save to Project Functions
        // =============================================
        var STORAGE_KEY = 'brandclave_saved_trends';

        function getSavedProjects() {
            try {
                var data = localStorage.getItem(STORAGE_KEY);
                return data ? JSON.parse(data) : [];
            } catch (e) {
                console.error('Error reading saved projects:', e);
                return [];
            }
        }

        function saveProject(trend) {
            var saved = getSavedProjects();
            for (var i = 0; i < saved.length; i++) { if (saved[i].id === trend.id) return false; }

            saved.push({
                id: trend.id,
                name: trend.name || trend.trend_name,
                description: trend.description,
                white_space_score: trend.white_space_score,
                strength_score: trend.strength_score,
                region: trend.region,
                audience_segment: trend.audience_segment,
                topics: trend.topics,
                saved_at: new Date().toISOString()
            });

            localStorage.setItem(STORAGE_KEY, JSON.stringify(saved));
            return true;
        }

        function removeProject(trendId) {
            var saved = getSavedProjects();
            var filtered = [];
            for (var i = 0; i < saved.length; i++) { if (saved[i].id !== trendId) filtered.push(saved[i]); }
            localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
        }

        function isProjectSaved(trendId) {
            var saved = getSavedProjects();
            for (var i = 0; i < saved.length; i++) { if (saved[i].id === trendId) return true; }
            return false;
        }

        function toggleSaveProject(idx) {
            var trend = allTrends[idx];
            if (!trend) return;

            if (isProjectSaved(trend.id)) {
                removeProject(trend.id);
            } else {
                saveProject(trend);
            }

            // Re-render the trends list
            var trendsHtml = '';
            for (var i = 0; i < allTrends.length; i++) { trendsHtml += renderTrend(allTrends[i], i); }
            document.getElementById('trends-list').innerHTML = trendsHtml;
            updateSavedCount();
            renderMyProjects(); // Auto-update My Projects tab
        }

        function updateSavedCount() {
            var count = getSavedProjects().length;
            var countEl = document.getElementById('saved-count');
            if (countEl) {
                countEl.textContent = count > 0 ? count + ' saved' : '';
            }
        }

        // =============================================
        // LocalStorage Save Moves Functions
        // =============================================
        var MOVES_STORAGE_KEY = 'brandclave_saved_moves';

        function getSavedMoves() {
            try {
                var data = localStorage.getItem(MOVES_STORAGE_KEY);
                return data ? JSON.parse(data) : [];
            } catch (e) {
                console.error('Error reading saved moves:', e);
                return [];
            }
        }

        function saveMove(move) {
            var saved = getSavedMoves();
            for (var i = 0; i < saved.length; i++) { if (saved[i].id === move.id) return false; }

            saved.push({
                id: move.id,
                title: move.title,
                summary: move.summary,
                company: move.company,
                move_type: move.move_type,
                market: move.market,
                strategic_implications: move.strategic_implications,
                source_name: move.source_name,
                saved_at: new Date().toISOString()
            });

            localStorage.setItem(MOVES_STORAGE_KEY, JSON.stringify(saved));
            return true;
        }

        function removeMove(moveId) {
            var saved = getSavedMoves();
            var filtered = [];
            for (var i = 0; i < saved.length; i++) { if (saved[i].id !== moveId) filtered.push(saved[i]); }
            localStorage.setItem(MOVES_STORAGE_KEY, JSON.stringify(filtered));
        }

        function isMoveSaved(moveId) {
            var saved = getSavedMoves();
            for (var i = 0; i < saved.length; i++) { if (saved[i].id === moveId) return true; }
            return false;
        }

        function toggleSaveMove(idx) {
            var move = allMoves[idx];
            if (!move) return;

            if (isMoveSaved(move.id)) {
                removeMove(move.id);
            } else {
                saveMove(move);
            }

            // Re-render the moves list
            var movesHtml = '';
            for (var i = 0; i < allMoves.length; i++) { movesHtml += renderMove(allMoves[i], i); }
            document.getElementById('moves-list').innerHTML = movesHtml;
            updateMovesSavedCount();
            renderMyProjects(); // Auto-update My Projects tab
        }

        function updateMovesSavedCount() {
            var count = getSavedMoves().length;
            var countEl = document.getElementById('moves-saved-count');
            if (countEl) {
                countEl.textContent = count > 0 ? count + ' saved' : '';
            }
        }

        // =============================================
        // My Projects Functions
        // =============================================
        function renderMyProjects() {
            var savedTrends = getSavedProjects();
            var savedMoves = getSavedMoves();

            // Update counts
            document.getElementById('saved-trends-count').textContent = savedTrends.length > 0 ? '(' + savedTrends.length + ')' : '';
            document.getElementById('saved-moves-count').textContent = savedMoves.length > 0 ? '(' + savedMoves.length + ')' : '';

            // Render saved trends
            var trendsListEl = document.getElementById('saved-trends-list');
            if (savedTrends.length > 0) {
                var trendsHtml = '';
                for (var i = 0; i < savedTrends.length; i++) {
                    var t = savedTrends[i];
                    var tMeta = (t.region ? t.region + ' • ' : '') + (t.audience_segment || 'General');
                    if (t.white_space_score) tMeta += ' • White Space: ' + (t.white_space_score * 100).toFixed(0) + '%';
                    trendsHtml += '<div class="saved-item-card"><div><h4>' + (t.name || 'Unnamed Trend') + '</h4><div class="saved-item-meta">' + tMeta + '</div></div><div class="saved-item-actions"><button class="btn-remove" onclick="removeSavedTrend('' + t.id + '')">Remove</button></div></div>';
                }
                trendsListEl.innerHTML = trendsHtml;
            } else {
                trendsListEl.innerHTML = '<div class="empty"><div class="icon">📊</div>No saved trends yet</div>';
            }

            // Render saved moves
            var movesListEl = document.getElementById('saved-moves-list');
            if (savedMoves.length > 0) {
                var movesHtml = '';
                for (var j = 0; j < savedMoves.length; j++) {
                    var m = savedMoves[j];
                    var mMeta = (m.company || 'Unknown') + ' • ' + (m.move_type ? m.move_type.replace('_', ' ') : 'Move');
                    if (m.market) mMeta += ' • ' + m.market;
                    movesHtml += '<div class="saved-item-card"><div><h4>' + (m.title || 'Unnamed Move') + '</h4><div class="saved-item-meta">' + mMeta + '</div></div><div class="saved-item-actions"><button class="btn-remove" onclick="removeSavedMove('' + m.id + '')">Remove</button></div></div>';
                }
                movesListEl.innerHTML = movesHtml;
            } else {
                movesListEl.innerHTML = '<div class="empty"><div class="icon">♟️</div>No saved moves yet</div>';
            }

            // Update profile insights
            updateProfileInsights(savedTrends, savedMoves);

            // Enable/disable build button
            var buildBtn = document.getElementById('build-from-profile-btn');
            buildBtn.disabled = (savedTrends.length + savedMoves.length) === 0;
        }

        function updateProfileInsights(trends, moves) {
            var profileEl = document.getElementById('profile-content');

            if (trends.length === 0 && moves.length === 0) {
                profileEl.innerHTML = '<div class="empty" style="color:rgba(255,255,255,0.8);"><div class="icon">💡</div>Save trends and moves to build your profile</div>';
                return;
            }

            // Analyze patterns
            var regions = {};
            var segments = {};
            var topics = {};
            var companies = {};
            var moveTypes = {};
            var markets = {};

            // From trends
            for (var ti = 0; ti < trends.length; ti++) {
                var t = trends[ti];
                if (t.region) regions[t.region] = (regions[t.region] || 0) + 1;
                if (t.audience_segment) segments[t.audience_segment] = (segments[t.audience_segment] || 0) + 1;
                var tTopics = t.topics || [];
                for (var tpi = 0; tpi < tTopics.length; tpi++) {
                    topics[tTopics[tpi]] = (topics[tTopics[tpi]] || 0) + 1;
                }
            }

            // From moves
            for (var mi = 0; mi < moves.length; mi++) {
                var m = moves[mi];
                if (m.company) companies[m.company] = (companies[m.company] || 0) + 1;
                if (m.move_type) moveTypes[m.move_type] = (moveTypes[m.move_type] || 0) + 1;
                if (m.market) markets[m.market] = (markets[m.market] || 0) + 1;
            }

            // Sort by frequency and take top items
            function sortByFreq(obj) {
                var entries = [];
                for (var k in obj) { if (obj.hasOwnProperty(k)) entries.push([k, obj[k]]); }
                entries.sort(function(a, b) { return b[1] - a[1]; });
                return entries;
            }
            var topRegions = sortByFreq(regions).slice(0, 3);
            var topSegments = sortByFreq(segments).slice(0, 3);
            var topTopics = sortByFreq(topics).slice(0, 5);
            var topCompanies = sortByFreq(companies).slice(0, 3);
            var topMoveTypes = sortByFreq(moveTypes).slice(0, 3);
            var topMarkets = sortByFreq(markets).slice(0, 3);

            var html = '';

            if (topRegions.length > 0 || topMarkets.length > 0) {
                var allLocations = topRegions.concat(topMarkets).slice(0, 4);
                var locHtml = '';
                for (var li = 0; li < allLocations.length; li++) { locHtml += '<span class="profile-tag">' + allLocations[li][0] + '</span>'; }
                html += '<div class="profile-section"><div class="profile-section-title">📍 Locations of Interest</div>' + locHtml + '</div>';
            }

            if (topSegments.length > 0) {
                var segHtml = '';
                for (var si = 0; si < topSegments.length; si++) { segHtml += '<span class="profile-tag">' + topSegments[si][0] + '</span>'; }
                html += '<div class="profile-section"><div class="profile-section-title">👥 Target Segments</div>' + segHtml + '</div>';
            }

            if (topTopics.length > 0) {
                var topHtml = '';
                for (var toi = 0; toi < topTopics.length; toi++) { topHtml += '<span class="profile-tag">' + topTopics[toi][0] + '</span>'; }
                html += '<div class="profile-section"><div class="profile-section-title">🔥 Key Themes</div>' + topHtml + '</div>';
            }

            if (topCompanies.length > 0) {
                var coHtml = '';
                for (var coi = 0; coi < topCompanies.length; coi++) { coHtml += '<span class="profile-tag">' + topCompanies[coi][0] + '</span>'; }
                html += '<div class="profile-section"><div class="profile-section-title">🏨 Companies Watched</div>' + coHtml + '</div>';
            }

            if (topMoveTypes.length > 0) {
                var mtHtml = '';
                for (var mti = 0; mti < topMoveTypes.length; mti++) { mtHtml += '<span class="profile-tag">' + topMoveTypes[mti][0].replace('_', ' ') + '</span>'; }
                html += '<div class="profile-section"><div class="profile-section-title">♟️ Move Types</div>' + mtHtml + '</div>';
            }

            profileEl.innerHTML = html || '<div style="opacity:0.8;">Collecting insights...</div>';
        }

        function removeSavedTrend(trendId) {
            removeProject(trendId);
            renderMyProjects();
            updateSavedCount();
            // Re-render trends if visible
            if (allTrends.length > 0) {
                var trendsHtml = '';
                for (var i = 0; i < allTrends.length; i++) { trendsHtml += renderTrend(allTrends[i], i); }
                document.getElementById('trends-list').innerHTML = trendsHtml;
            }
        }

        function removeSavedMove(moveId) {
            removeMove(moveId);
            renderMyProjects();
            updateMovesSavedCount();
            // Re-render moves if visible
            if (allMoves.length > 0) {
                var movesHtml = '';
                for (var i = 0; i < allMoves.length; i++) { movesHtml += renderMove(allMoves[i], i); }
                document.getElementById('moves-list').innerHTML = movesHtml;
            }
        }

        function clearAllSaved() {
            if (!confirm('Are you sure you want to clear all saved items?')) return;
            localStorage.removeItem(STORAGE_KEY);
            localStorage.removeItem(MOVES_STORAGE_KEY);
            renderMyProjects();
            updateSavedCount();
            updateMovesSavedCount();
            // Re-render lists
            if (allTrends.length > 0) {
                var trendsHtml = '';
                for (var i = 0; i < allTrends.length; i++) { trendsHtml += renderTrend(allTrends[i], i); }
                document.getElementById('trends-list').innerHTML = trendsHtml;
            }
            if (allMoves.length > 0) {
                var movesHtml = '';
                for (var j = 0; j < allMoves.length; j++) { movesHtml += renderMove(allMoves[j], j); }
                document.getElementById('moves-list').innerHTML = movesHtml;
            }
        }

        function buildBrandFromProfile() {
            var savedTrends = getSavedProjects();
            var savedMoves = getSavedMoves();

            if (savedTrends.length === 0 && savedMoves.length === 0) {
                alert('Save some trends or moves first to build your profile.');
                return;
            }

            // Helper to get unique values
            function getUnique(arr) {
                var seen = {};
                var result = [];
                for (var i = 0; i < arr.length; i++) {
                    if (arr[i] && !seen[arr[i]]) {
                        seen[arr[i]] = true;
                        result.push(arr[i]);
                    }
                }
                return result;
            }

            // Extract values from trends
            var trendRegions = [];
            var trendSegments = [];
            var trendTopics = [];
            for (var ti = 0; ti < savedTrends.length; ti++) {
                if (savedTrends[ti].region) trendRegions.push(savedTrends[ti].region);
                if (savedTrends[ti].audience_segment) trendSegments.push(savedTrends[ti].audience_segment);
                var topics = savedTrends[ti].topics || [];
                for (var tpi = 0; tpi < topics.length; tpi++) { trendTopics.push(topics[tpi]); }
            }

            // Extract values from moves
            var moveCompanies = [];
            var moveMarkets = [];
            var moveTypes = [];
            for (var mi = 0; mi < savedMoves.length; mi++) {
                if (savedMoves[mi].company) moveCompanies.push(savedMoves[mi].company);
                if (savedMoves[mi].market) moveMarkets.push(savedMoves[mi].market);
                if (savedMoves[mi].move_type) moveTypes.push(savedMoves[mi].move_type);
            }

            // Build profile data for brand generation
            var profileData = {
                trends: savedTrends,
                moves: savedMoves,
                regions: getUnique(trendRegions),
                segments: getUnique(trendSegments),
                topics: getUnique(trendTopics),
                companies: getUnique(moveCompanies),
                markets: getUnique(moveMarkets),
                move_types: getUnique(moveTypes)
            };

            // Store for Build a Brand page
            sessionStorage.setItem('brandclave_profile_data', JSON.stringify(profileData));
            sessionStorage.setItem('brandclave_brand_input', JSON.stringify({
                from_profile: true,
                initial_region: profileData.regions[0] || '',
                initial_segment: profileData.segments[0] || 'lifestyle',
                topics: profileData.topics.slice(0, 5)
            }));

            // Navigate to Build a Brand
            window.location.href = '/api/monitoring/build-a-brand';
        }

        // =============================================
        // Turn Into Brand Function
        // =============================================
        function turnIntoBrand(idx) {
            var trend = allTrends[idx];
            if (!trend) return;

            // Store trend data for the Build a Brand page
            var brandInput = {
                source_trend_id: trend.id,
                source_trend_name: trend.name || trend.trend_name,
                initial_segment: trend.audience_segment || 'lifestyle',
                initial_region: trend.region || '',
                topics: trend.topics || [],
                white_space_score: trend.white_space_score,
                description: trend.description,
                why_it_matters: trend.why_it_matters
            };

            // Store in sessionStorage
            sessionStorage.setItem('brandclave_brand_input', JSON.stringify(brandInput));

            // Navigate to Build a Brand page
            window.location.href = '/api/monitoring/build-a-brand';
        }

        // City Desires functions
        function quickCity(city, country) {
            document.getElementById('city-input').value = city;
            document.getElementById('country-input').value = country;
            analyzeCity();
        }

        // Event delegation for desire quotes interactions
        document.addEventListener('click', function(e) {
            // Handle source badge clicks
            if (e.target.classList.contains('source-badge')) {
                var cardId = e.target.getAttribute('data-card');
                var sourceName = e.target.getAttribute('data-source');
                filterDesireQuotes(cardId, sourceName);
            }
            // Handle expand button clicks
            if (e.target.classList.contains('expand-btn')) {
                var cardId = e.target.getAttribute('data-card');
                var moreCount = parseInt(e.target.getAttribute('data-count'));
                toggleDesireQuotes(cardId, moreCount, e.target);
            }
            // Handle show all button clicks
            if (e.target.classList.contains('show-all-btn')) {
                var cardId = e.target.getAttribute('data-card');
                filterDesireQuotes(cardId, 'all');
            }
        });

        // Toggle expand/collapse for desire quotes
        function toggleDesireQuotes(cardId, moreCount, btn) {
            var card = document.getElementById(cardId);
            if (!card) return;

            var allQuotes = card.querySelectorAll('.desire-quote');
            var hiddenQuotes = [];
            for (var i = 0; i < allQuotes.length; i++) {
                if (allQuotes[i].style.display === 'none') {
                    hiddenQuotes.push(allQuotes[i]);
                }
            }
            var isExpanded = hiddenQuotes.length === 0;

            if (isExpanded) {
                // Collapse: hide quotes beyond initial 2
                for (var i = 0; i < allQuotes.length; i++) {
                    if (i >= 2) allQuotes[i].style.display = 'none';
                }
                if (btn) btn.textContent = 'Show ' + moreCount + ' more quotes';
            } else {
                // Expand: show all quotes
                for (var i = 0; i < hiddenQuotes.length; i++) {
                    hiddenQuotes[i].style.display = 'block';
                }
                if (btn) btn.textContent = 'Show less';
            }
        }

        // Filter quotes by source when clicking source badge
        function filterDesireQuotes(cardId, sourceName) {
            var card = document.getElementById(cardId);
            if (!card) return;

            var allQuotes = card.querySelectorAll('.desire-quote');
            var expandBtn = card.querySelector('.expand-btn');

            if (sourceName === 'all') {
                // Show first 2, hide rest (reset to initial state)
                for (var i = 0; i < allQuotes.length; i++) {
                    allQuotes[i].style.display = i < 2 ? 'block' : 'none';
                }
                if (expandBtn) {
                    var hiddenCount = Math.max(0, allQuotes.length - 2);
                    expandBtn.style.display = hiddenCount > 0 ? 'block' : 'none';
                    expandBtn.textContent = 'Show ' + hiddenCount + ' more quotes';
                }
            } else {
                // Filter by source - show all matching, hide expand button
                for (var i = 0; i < allQuotes.length; i++) {
                    var qSource = allQuotes[i].getAttribute('data-source');
                    if (qSource === sourceName) {
                        allQuotes[i].style.display = 'block';
                    } else {
                        allQuotes[i].style.display = 'none';
                    }
                }
                if (expandBtn) {
                    expandBtn.style.display = 'none';
                }
            }
        }

        async function analyzeCity() {
            var city = document.getElementById('city-input').value.trim();
            var country = document.getElementById('country-input').value.trim();

            if (!city) {
                alert('Please enter a city name');
                return;
            }

            var btn = document.getElementById('analyze-btn');
            var resultsDiv = document.getElementById('city-results');

            btn.disabled = true;
            btn.textContent = 'Analyzing...';
            resultsDiv.innerHTML = '<div class="empty"><div class="icon">⏳</div>Analyzing ' + city + '... This may take 60-120 seconds (using semantic clustering).</div>';

            try {
                // Use adaptive endpoint with semantic clustering for better results
                var response = await fetch('/api/city-desires/adaptive', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ city: city, country: country })
                });

                if (!response.ok) {
                    throw new Error('Analysis failed: ' + response.status);
                }

                var data = await response.json();
                renderCityResults(data);

            } catch (err) {
                resultsDiv.innerHTML = '<div class="error">Analysis failed: ' + err.message + '</div>';
            } finally {
                btn.disabled = false;
                btn.textContent = 'Analyze';
            }
        }

        function renderCityResults(data) {
            var resultsDiv = document.getElementById('city-results');

            // Format sources summary
            var sourcesSummary = data.sources_summary || {};
            var sourcesEntries = [];
            for (var k in sourcesSummary) { if (sourcesSummary.hasOwnProperty(k)) sourcesEntries.push([k, sourcesSummary[k]]); }
            sourcesEntries.sort(function(a, b) { return b[1] - a[1]; });
            var sourcesHtml = sourcesEntries.map(function(entry) { return '<span style="background:#e8f4fd;padding:3px 8px;border-radius:12px;font-size:0.85em;margin-right:6px;">' + entry[0] + ': ' + entry[1] + '</span>'; }).join('');

            var html = '<div style="background:#f8f9fa;padding:15px;border-radius:8px;margin-bottom:20px;">' +
                '<h3 style="margin-bottom:10px;">' + data.city + ', ' + data.country + '</h3>' +
                '<div style="display:flex;gap:20px;flex-wrap:wrap;margin-bottom:10px;">' +
                '<div><strong>' + (data.total_signals || 0) + '</strong> signals</div>' +
                '<div><strong>' + (data.num_learned_categories || 0) + '</strong> themes discovered</div>' +
                '<div>Confidence: <strong>' + ((data.model_confidence || 0) * 100).toFixed(0) + '%</strong></div>' +
                '</div>' +
                (sourcesHtml ? '<div style="margin-top:10px;">Sources: ' + sourcesHtml + '</div>' : '') +
                '</div>';

            // Top Desires with source attribution
            if (data.top_desires && data.top_desires.length > 0) {
                html += '<h3 style="margin:20px 0 10px;">🔥 Top Desires</h3>';
                html += data.top_desires.slice(0, 5).map(function(d, idx) {
                    // Format sources as visual badges
                    var sourceColors = {
                        'reddit': '#ff4500',
                        'youtube': '#ff0000',
                        'tripadvisor': '#00af87',
                        'twitter': '#1da1f2',
                        'instagram': '#e1306c'
                    };
                    var cardId = 'desire-card-' + idx;
                    var sourceBadges = (d.sources || [])
                        .map(function(s) {
                            var color = sourceColors[s.name.toLowerCase()] || '#666';
                            var sourceName = s.name.toLowerCase();
                            return '<span class="source-badge" data-card="' + cardId + '" data-source="' + sourceName + '" style="display:inline-block;background:' + color + ';color:white;padding:2px 8px;border-radius:12px;font-size:0.75em;margin-right:4px;cursor:pointer;">' + s.name + ' (' + s.count + ')</span>';
                        })
                        .join('') || '<span style="color:#999;font-size:0.85em;">No source data</span>';

                    // Get all example snippets for expandable section
                    var examplesHtml = '';
                    if (d.example_snippets && d.example_snippets.length > 0) {
                        var allSnippets = d.example_snippets.slice(0, 10);
                        var initialCount = 2;

                        function renderQuote(snippet, hidden) {
                            var text = typeof snippet === 'string' ? snippet : (snippet.text || '');
                            var source = typeof snippet === 'object' ? snippet.source : 'traveler';
                            var sourceColor = sourceColors[source.toLowerCase()] || '#666';
                            if (text) {
                                var hiddenStyle = hidden ? 'display:none;' : '';
                                return '<div class="desire-quote" data-source="' + source.toLowerCase() + '" style="' + hiddenStyle + 'margin-bottom:8px;padding:8px 10px;background:rgba(0,0,0,0.02);border-radius:4px;font-size:0.85em;border-left:3px solid ' + sourceColor + ';"><span style="font-style:italic;">"' + text.substring(0, 180) + '..."</span> <span style="color:' + sourceColor + ';font-weight:500;">- ' + source + '</span></div>';
                            }
                            return '';
                        }

                        // Initial visible quotes
                        var visibleQuotes = allSnippets.slice(0, initialCount).map(function(s) { return renderQuote(s, false); }).filter(function(h) { return h; }).join('');

                        // Hidden quotes (for expansion)
                        var hiddenQuotes = allSnippets.slice(initialCount).map(function(s) { return renderQuote(s, true); }).filter(function(h) { return h; }).join('');

                        var hasMore = allSnippets.length > initialCount;
                        var moreCount = allSnippets.length - initialCount;

                        if (visibleQuotes || hiddenQuotes) {
                            var expandBtnHtml = '';
                            if (hasMore) {
                                expandBtnHtml = '<div class="expand-btn" data-card="' + cardId + '" data-count="' + moreCount + '" style="margin-top:8px;padding:6px 12px;background:#f5f5f5;border-radius:4px;font-size:0.8em;color:#4a90a4;cursor:pointer;text-align:center;">Show ' + moreCount + ' more quotes</div>';
                            }
                            examplesHtml = '<div style="margin-top:12px;" id="' + cardId + '-quotes">' +
                                '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">' +
                                '<span style="font-size:0.8em;color:#666;text-transform:uppercase;letter-spacing:0.5px;">What travelers are saying:</span>' +
                                '<span class="show-all-btn" data-card="' + cardId + '" style="font-size:0.75em;color:#4a90a4;cursor:pointer;">Show all</span>' +
                                '</div>' +
                                '<div class="quotes-container">' + visibleQuotes + hiddenQuotes + '</div>' +
                                expandBtnHtml +
                                '</div>';
                        }
                    }

                    // Build insights section
                    var insightsHtml = '';
                    if (d.unmet_need) {
                        insightsHtml += '<div style="margin-top:10px;"><strong style="color:#d32f2f;">Unmet Need:</strong> ' + d.unmet_need + '</div>';
                    }
                    if (d.why_supply_fails) {
                        insightsHtml += '<div style="margin-top:6px;"><strong style="color:#f57c00;">Why Supply Fails:</strong> ' + d.why_supply_fails + '</div>';
                    }
                    if (d.solving_features && d.solving_features.length > 0) {
                        insightsHtml += '<div style="margin-top:6px;"><strong style="color:#388e3c;">What Would Solve This:</strong> ' + d.solving_features.slice(0, 3).join(' • ') + '</div>';
                    }
                    if (d.target_guest) {
                        insightsHtml += '<div style="margin-top:6px;"><strong style="color:#1976d2;">Target Guest:</strong> ' + d.target_guest + '</div>';
                    }

                    // Fallback to description if no structured insights
                    if (!insightsHtml && d.description) {
                        insightsHtml = '<div style="margin-top:8px;">' + d.description + '</div>';
                    }

                    return '<div class="desire-card" id="' + cardId + '">' +
                        '<div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:8px;">' +
                        '<h4 style="margin:0;">' + (d.theme_name || d.theme || 'Desire') + '</h4>' +
                        '<div>' + sourceBadges + '</div>' +
                        '</div>' +
                        insightsHtml +
                        '<div class="desire-meta" style="margin-top:12px;padding-top:10px;border-top:1px solid #eee;font-size:0.9em;color:#666;">' +
                        '<strong>' + (d.frequency || 0) + '</strong> mentions • ' +
                        '<strong>' + ((d.intensity_score || 0) * 100).toFixed(0) + '%</strong> intensity' +
                        '</div>' +
                        examplesHtml +
                        '</div>';
                }).join('');
            }

            // White Space Opportunities
            if (data.white_space_opportunities && data.white_space_opportunities.length > 0) {
                html += '<h3 style="margin:20px 0 10px;">💡 White Space Opportunities</h3>';
                html += data.white_space_opportunities.slice(0, 5).map(function(o) {
                    return '<div class="opportunity-card">' + o + '</div>';
                }).join('');
            }

            // Concept Lanes
            if (data.concept_lanes && data.concept_lanes.length > 0) {
                html += '<h3 style="margin:20px 0 10px;">🎯 Concept Lanes</h3>';
                html += data.concept_lanes.slice(0, 3).map(function(c) {
                    var name = c.name || c.concept || 'Hotel Concept';
                    var positioning = c.positioning || c.rationale || '';
                    var solves = c.solves || '';
                    var differentiators = c.key_differentiators || c.key_features || [];
                    var targetGuest = c.target_guest || '';
                    var pricePosition = c.price_position || '';
                    var whyWins = c.why_it_wins || '';

                    var detailsHtml = '';
                    if (positioning) {
                        detailsHtml += '<div style="margin-top:8px;font-style:italic;color:#555;">' + positioning + '</div>';
                    }
                    if (solves) {
                        detailsHtml += '<div style="margin-top:8px;"><strong>Solves:</strong> ' + solves + '</div>';
                    }
                    if (differentiators.length > 0) {
                        detailsHtml += '<div style="margin-top:8px;"><strong>Key Differentiators:</strong> ' + differentiators.slice(0, 3).join(' • ') + '</div>';
                    }
                    if (targetGuest) {
                        detailsHtml += '<div style="margin-top:6px;"><strong>Target:</strong> ' + targetGuest + '</div>';
                    }
                    if (pricePosition) {
                        detailsHtml += '<span style="display:inline-block;margin-top:8px;background:#e8f4fd;padding:3px 10px;border-radius:12px;font-size:0.85em;">' + pricePosition + '</span>';
                    }
                    if (whyWins) {
                        detailsHtml += '<div style="margin-top:10px;padding:8px;background:#f0f8e8;border-radius:4px;font-size:0.9em;"><strong>Why it wins:</strong> ' + whyWins + '</div>';
                    }

                    return '<div class="concept-card"><h4>' + name + '</h4>' + detailsHtml + '</div>';
                }).join('');
            }

            // Underserved Segments
            if (data.underserved_segments && data.underserved_segments.length > 0) {
                html += '<h3 style="margin:20px 0 10px;">👥 Underserved Segments</h3>';
                html += '<div style="display:flex;flex-wrap:wrap;gap:8px;">';
                html += data.underserved_segments.map(function(s) {
                    return '<span style="background:#e0e0e0;padding:5px 12px;border-radius:20px;font-size:0.9em;">' + s + '</span>';
                }).join('');
                html += '</div>';
            }

            resultsDiv.innerHTML = html;
        }

        // Chat functionality
        var chatConversationId = null;

        function buildProfileContext() {
            var savedTrends = getSavedProjects();
            var savedMoves = getSavedMoves();

            if (savedTrends.length === 0 && savedMoves.length === 0) {
                return null;
            }

            // Helper function to get unique values
            function getUniqueVals(arr) {
                var seen = {};
                var result = [];
                for (var i = 0; i < arr.length; i++) {
                    if (arr[i] && !seen[arr[i]]) {
                        seen[arr[i]] = true;
                        result.push(arr[i]);
                    }
                }
                return result;
            }

            var context = 'User research profile: ';

            // Trends summary
            if (savedTrends.length > 0) {
                var trendNames = [];
                var trendRegions = [];
                var trendSegments = [];
                var trendTopics = [];
                for (var i = 0; i < savedTrends.length && i < 3; i++) {
                    if (savedTrends[i].name) trendNames.push(savedTrends[i].name);
                }
                for (var j = 0; j < savedTrends.length; j++) {
                    if (savedTrends[j].region) trendRegions.push(savedTrends[j].region);
                    if (savedTrends[j].audience_segment) trendSegments.push(savedTrends[j].audience_segment);
                    var topics = savedTrends[j].topics || [];
                    for (var k = 0; k < topics.length; k++) { trendTopics.push(topics[k]); }
                }
                var regions = getUniqueVals(trendRegions);
                var segments = getUniqueVals(trendSegments);
                var topics = getUniqueVals(trendTopics).slice(0, 5);

                context += 'Tracking ' + savedTrends.length + ' trends';
                if (trendNames.length) context += ' including "' + trendNames.join('", "') + '"';
                if (regions.length) context += '. Interested in regions: ' + regions.join(', ');
                if (segments.length) context += '. Target segments: ' + segments.join(', ');
                if (topics.length) context += '. Key themes: ' + topics.join(', ');
                context += '. ';
            }

            // Moves summary
            if (savedMoves.length > 0) {
                var moveCompanies = [];
                var moveMarkets = [];
                var moveMoveTypes = [];
                for (var mi = 0; mi < savedMoves.length; mi++) {
                    if (savedMoves[mi].company) moveCompanies.push(savedMoves[mi].company);
                    if (savedMoves[mi].market) moveMarkets.push(savedMoves[mi].market);
                    if (savedMoves[mi].move_type) moveMoveTypes.push(savedMoves[mi].move_type);
                }
                var companies = getUniqueVals(moveCompanies).slice(0, 3);
                var markets = getUniqueVals(moveMarkets).slice(0, 3);
                var moveTypes = getUniqueVals(moveMoveTypes);

                context += 'Watching ' + savedMoves.length + ' strategic moves';
                if (companies.length) context += ' by companies like ' + companies.join(', ');
                if (markets.length) context += ' in markets: ' + markets.join(', ');
                if (moveTypes.length) {
                    var mtFormatted = [];
                    for (var mti = 0; mti < moveTypes.length; mti++) { mtFormatted.push(moveTypes[mti].replace('_', ' ')); }
                    context += '. Move types: ' + mtFormatted.join(', ');
                }
                context += '.';
            }

            return context;
        }

        function sendSuggestion(text) {
            document.getElementById('chat-input').value = text;
            sendMessage();
        }

        function startBrandBuild() {
            var savedTrends = getSavedProjects();
            var savedMoves = getSavedMoves();

            // Helper function
            function getUniqueVals(arr) {
                var seen = {};
                var result = [];
                for (var i = 0; i < arr.length; i++) {
                    if (arr[i] && !seen[arr[i]]) {
                        seen[arr[i]] = true;
                        result.push(arr[i]);
                    }
                }
                return result;
            }

            var message;

            if (savedTrends.length > 0 || savedMoves.length > 0) {
                // Has profile - build from it
                var trendRegions = [];
                var trendSegments = [];
                var trendTopics = [];
                for (var i = 0; i < savedTrends.length; i++) {
                    if (savedTrends[i].region) trendRegions.push(savedTrends[i].region);
                    if (savedTrends[i].audience_segment) trendSegments.push(savedTrends[i].audience_segment);
                    var topics = savedTrends[i].topics || [];
                    for (var j = 0; j < topics.length; j++) { trendTopics.push(topics[j]); }
                }
                var regions = getUniqueVals(trendRegions);
                var segments = getUniqueVals(trendSegments);
                var topicsArr = getUniqueVals(trendTopics).slice(0, 3);

                message = 'Help me build a hotel brand based on my research profile.';
                if (regions.length) message += ' I am interested in ' + regions.slice(0, 2).join(' and ') + '.';
                if (segments.length) message += ' Target segment: ' + segments[0] + '.';
                if (topicsArr.length) message += ' Key themes I have been tracking: ' + topicsArr.join(', ') + '.';
            } else {
                // No profile - ask for guidance
                message = 'I want to build a hotel brand but I am not sure where to start. Can you help me figure out what kind of brand would be right?';
            }

            document.getElementById('chat-input').value = message;
            sendMessage();
        }

        async function sendMessage() {
            var input = document.getElementById('chat-input');
            var message = input.value.trim();
            if (!message) return;

            var messagesDiv = document.getElementById('chat-messages');

            // Clear welcome message if present
            var welcome = messagesDiv.querySelector('.chat-welcome');
            if (welcome) welcome.remove();

            // Add user message
            messagesDiv.innerHTML += '<div class="chat-message user"><div class="chat-bubble">' + escapeHtml(message) + '</div></div>';
            input.value = '';

            // Add typing indicator
            messagesDiv.innerHTML += '<div class="chat-message assistant" id="typing-indicator"><div class="chat-bubble chat-typing"><span></span><span></span><span></span></div></div>';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;

            // Build profile context from saved items
            var profileContext = buildProfileContext();

            try {
                var res = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        message: message,
                        conversation_id: chatConversationId,
                        user_context: profileContext
                    })
                });

                var data = await res.json();

                // Remove typing indicator
                var typing = document.getElementById('typing-indicator');
                if (typing) typing.remove();

                if (res.ok) {
                    chatConversationId = data.conversation_id;

                    // Add assistant message
                    var confClass = data.confidence === 'High' ? 'confidence-high' :
                                      data.confidence === 'Medium' ? 'confidence-medium' : 'confidence-low';
                    messagesDiv.innerHTML += '<div class="chat-message assistant"><div class="chat-bubble">' + formatResponse(data.response) + '</div><div class="chat-confidence ' + confClass + '">' + data.confidence + ' confidence | ' + data.sources_used + ' sources | Mode: ' + data.mode + '</div></div>';

                    // Show state info
                    if (data.state) {
                        var slotLoc = (data.state.slots && data.state.slots.location) ? data.state.slots.location : '-';
                        var slotSeg = (data.state.slots && data.state.slots.segment) ? data.state.slots.segment : '-';
                        document.getElementById('chat-state').innerHTML = 'Mode: ' + data.mode + ' (' + Math.round(data.state.mode_confidence * 100) + '%) | Location: ' + slotLoc + ' | Segment: ' + slotSeg;
                    }

                    // Show suggested action
                    if (data.suggested_action) {
                        messagesDiv.innerHTML += '<div class="chat-message assistant"><button onclick="window.location.href='/api/monitoring/dashboard-v2#build'" class="suggestion-chip" style="margin-top:10px;">➡️ Continue to Build a Brand</button></div>';
                    }
                } else {
                    messagesDiv.innerHTML += '<div class="chat-message assistant"><div class="chat-bubble" style="background:#fee2e2;color:#991b1b;">Error: ' + (data.detail || 'Something went wrong') + '</div></div>';
                }

            } catch (err) {
                var typing = document.getElementById('typing-indicator');
                if (typing) typing.remove();
                messagesDiv.innerHTML += '<div class="chat-message assistant"><div class="chat-bubble" style="background:#fee2e2;color:#991b1b;">Connection error: ' + err.message + '</div></div>';
            }

            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function escapeHtml(text) {
            var div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function formatResponse(text) {
            // Basic markdown-like formatting
            return escapeHtml(text)
                .replace(/\n/g, '<br>')
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                .replace(/\*(.*?)\*/g, '<em>$1</em>');
        }

        // Load data on page load
        loadAllData();
        loadFilterOptions();
        loadMoveFilterOptions();
        updateSavedCount();
        updateMovesSavedCount();
        renderMyProjects();

        // Auto-refresh every 60 seconds
        setInterval(loadAllData, 60000);
    