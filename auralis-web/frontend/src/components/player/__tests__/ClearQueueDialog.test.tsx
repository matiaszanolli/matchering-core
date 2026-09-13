/**
 * ClearQueueDialog Component Tests
 *
 * Tests for confirmation flow, button callbacks, keyboard handling,
 * and accessibility attributes.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import userEvent from '@testing-library/user-event';
import { render, screen } from '@/test/test-utils';
import { ClearQueueDialog } from '../QueuePanel/ClearQueueDialog';

describe('ClearQueueDialog', () => {
  const defaultProps = {
    onConfirm: vi.fn(),
    onCancel: vi.fn(),
  };

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the dialog with title and buttons', () => {
    render(<ClearQueueDialog {...defaultProps} />);

    expect(screen.getByText('Clear the entire queue?')).toBeInTheDocument();
    expect(screen.getByText('Cancel')).toBeInTheDocument();
    expect(screen.getByText('Clear')).toBeInTheDocument();
  });

  it('calls onConfirm when Clear button is clicked', async () => {
    const user = userEvent.setup();
    render(<ClearQueueDialog {...defaultProps} />);

    await user.click(screen.getByText('Clear'));
    expect(defaultProps.onConfirm).toHaveBeenCalledTimes(1);
  });

  it('calls onCancel when Cancel button is clicked', async () => {
    const user = userEvent.setup();
    render(<ClearQueueDialog {...defaultProps} />);

    await user.click(screen.getByText('Cancel'));
    expect(defaultProps.onCancel).toHaveBeenCalledTimes(1);
  });

  it('calls onCancel when backdrop is clicked', async () => {
    const user = userEvent.setup();
    render(<ClearQueueDialog {...defaultProps} />);

    // The backdrop is the outermost div with the fixed overlay
    const backdrop = screen.getByRole('dialog').parentElement!;
    await user.click(backdrop);
    expect(defaultProps.onCancel).toHaveBeenCalledTimes(1);
  });

  it('does not call onCancel when dialog content is clicked', async () => {
    const user = userEvent.setup();
    render(<ClearQueueDialog {...defaultProps} />);

    await user.click(screen.getByRole('dialog'));
    expect(defaultProps.onCancel).not.toHaveBeenCalled();
  });

  it('calls onCancel when Escape key is pressed', async () => {
    const user = userEvent.setup();
    render(<ClearQueueDialog {...defaultProps} />);

    // A real keypress lands on the focused element and bubbles to the dialog's
    // handler — so put focus inside the dialog first, as a user tabbing to it
    // would, rather than dispatching keydown on the dialog node directly.
    screen.getByText('Cancel').focus();
    await user.keyboard('{Escape}');
    expect(defaultProps.onCancel).toHaveBeenCalledTimes(1);
  });

  it('has correct accessibility attributes', () => {
    render(<ClearQueueDialog {...defaultProps} />);

    const dialog = screen.getByRole('dialog');
    expect(dialog).toHaveAttribute('aria-modal', 'true');
    expect(dialog).toHaveAttribute('aria-labelledby', 'clear-queue-dialog-title');
  });

  it('uses an h2 heading for the title', () => {
    render(<ClearQueueDialog {...defaultProps} />);

    const heading = screen.getByText('Clear the entire queue?');
    expect(heading.tagName).toBe('H2');
  });
});
