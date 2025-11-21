import React, { useEffect, useRef } from 'react';
import mirador from 'mirador';

const DocumentDetailView = ({ docId, manifestUrl, onClose }) => {
    const miradorInstanceRef = useRef(null);

    useEffect(() => {
        const config = {
            id: 'mirador-viewer',
            windows: [
                {
                    manifestId: manifestUrl,
                    view: 'single', // or 'gallery'
                },
            ],
            window: {
                allowClose: false,
                allowMaximize: false,
                defaultSideBarPanel: 'info',
                sideBarOpenByDefault: true,
                hideWindowTitle: true, // Clean look
            },
            workspace: {
                showZoomControls: true,
                type: 'mosaic', // or 'elastic'
            },
            workspaceControlPanel: {
                enabled: false, // Hide the workspace add button
            },
        };

        miradorInstanceRef.current = mirador.viewer(config);

        return () => {
            // Cleanup if Mirador provides a destroy method, though it usually doesn't need explicit cleanup in this way
            // But we might want to unmount the component cleanly
            if (miradorInstanceRef.current) {
                // miradorInstanceRef.current.unmount(); // Hypothetical cleanup
            }
        };
    }, [manifestUrl]);

    return (
        <div className="document-detail-view" style={{ position: 'relative', height: '100%', width: '100%' }}>
            <div id="mirador-viewer" style={{ position: 'relative', width: '100%', height: '600px' }} />
        </div>
    );
};

export default DocumentDetailView;
