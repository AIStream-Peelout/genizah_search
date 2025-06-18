import React from 'react';

const ErrorMessage = ({ error, onDismiss }) => (
    <div className={`mb-6 rounded-xl p-4 ${error.type === 'rate_limit' ? 'bg-red-50 border border-red-200' : 'bg-orange-50 border border-orange-200'}`}>
        <div className="flex justify-between items-start">
            <div>
                <h4 className={`font-medium ${error.type === 'rate_limit' ? 'text-red-800' : 'text-orange-800'}`}>
                    {error.type === 'rate_limit' ? 'Rate Limit Exceeded' : 'Error'}
                </h4>
                <p className={`mt-1 ${error.type === 'rate_limit' ? 'text-red-700' : 'text-orange-700'}`}>
                    {error.message}
                </p>
            </div>
            <button
                onClick={onDismiss}
                className={`text-sm px-3 py-1 rounded-md ${error.type === 'rate_limit' ? 'bg-red-100 text-red-800 hover:bg-red-200' : 'bg-orange-100 text-orange-800 hover:bg-orange-200'}`}
            >
                Dismiss
            </button>
        </div>
    </div>
);

export default ErrorMessage;
// document message