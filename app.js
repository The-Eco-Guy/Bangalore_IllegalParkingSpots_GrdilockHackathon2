/**
 * Developer-centric UI Interactions
 * - Mobile Fullscreen Navigation
 * - Event Stream Table Selection & Timeline Progress Sync
 * - Filter Modes (All / Errors / Slow Spans)
 * - Copy Trace ID
 */

document.addEventListener('DOMContentLoaded', () => {
  // 1. Mobile Menu Toggle
  const mobileMenuBtn = document.getElementById('mobileMenuToggle');
  const closeMobileNav = document.getElementById('closeMobileNav');
  const mobileNavOverlay = document.getElementById('mobileNavOverlay');
  const mobileNavLinks = document.querySelectorAll('.mobile-nav-links a');

  if (mobileMenuBtn && mobileNavOverlay) {
    mobileMenuBtn.addEventListener('click', () => {
      mobileNavOverlay.classList.add('open');
      document.body.style.overflow = 'hidden';
    });
  }

  const closeMenu = () => {
    if (mobileNavOverlay) {
      mobileNavOverlay.classList.remove('open');
      document.body.style.overflow = '';
    }
  };

  if (closeMobileNav) {
    closeMobileNav.addEventListener('click', closeMenu);
  }

  mobileNavLinks.forEach(link => {
    link.addEventListener('click', closeMenu);
  });

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeMenu();
  });

  // 2. Event Stream Left Sidebar & Table Row Sync
  const eventItems = document.querySelectorAll('.event-id-item');
  const tableRows = document.querySelectorAll('.event-table tbody tr');

  eventItems.forEach(item => {
    item.addEventListener('click', () => {
      const targetId = item.getAttribute('data-target-id');
      
      // Update sidebar active states
      eventItems.forEach(i => i.classList.remove('active'));
      item.classList.add('active');

      // Update table row active highlight
      tableRows.forEach(row => {
        const rowId = row.getAttribute('data-row-id');
        const timelineBar = row.querySelector('.timeline-bar');
        
        if (rowId === targetId) {
          row.classList.add('active-row');
          if (timelineBar) timelineBar.classList.add('focused');
          row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
          row.classList.remove('active-row');
          if (timelineBar) timelineBar.classList.remove('focused');
        }
      });
    });
  });

  // Table row click to sync sidebar
  tableRows.forEach(row => {
    row.addEventListener('click', () => {
      const rowId = row.getAttribute('data-row-id');
      tableRows.forEach(r => {
        r.classList.remove('active-row');
        const bar = r.querySelector('.timeline-bar');
        if (bar) bar.classList.remove('focused');
      });
      row.classList.add('active-row');
      const bar = row.querySelector('.timeline-bar');
      if (bar) bar.classList.add('focused');

      eventItems.forEach(i => {
        if (i.getAttribute('data-target-id') === rowId) {
          i.classList.add('active');
          i.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
          i.classList.remove('active');
        }
      });
    });
  });

  // 3. Copy Trace ID functionality
  const copyTraceBtn = document.getElementById('copyTraceBtn');
  if (copyTraceBtn) {
    copyTraceBtn.addEventListener('click', () => {
      const traceText = document.getElementById('traceIdValue')?.innerText || '9f82d1c0a7e4b2e88a31';
      navigator.clipboard.writeText(traceText).then(() => {
        const originalHtml = copyTraceBtn.innerHTML;
        copyTraceBtn.innerHTML = '<span class="v-chip-dot pass"></span>COPIED';
        setTimeout(() => {
          copyTraceBtn.innerHTML = originalHtml;
        }, 1500);
      });
    });
  }

  // 4. Trace Filter Buttons (All / Errors / Latency)
  const filterBtns = document.querySelectorAll('.filter-btn');
  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const filter = btn.getAttribute('data-filter');

      tableRows.forEach(row => {
        const isError = row.classList.contains('row-error');
        const isSlow = row.classList.contains('row-slow');

        if (filter === 'all') {
          row.style.display = '';
        } else if (filter === 'errors') {
          row.style.display = isError ? '' : 'none';
        } else if (filter === 'slow') {
          row.style.display = isSlow ? '' : 'none';
        }
      });
    });
  });
});
